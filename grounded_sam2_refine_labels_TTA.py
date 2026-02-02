import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict


import torchvision.transforms.functional as TF
from ensemble_boxes import weighted_boxes_fusion, soft_nms


SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""
Hyper parameters
"""
DUMP_JSON_RESULTS = True
TEXT_PROMPT = "spacecraft"

split_name = 'train_sunlamp_500'
gt_json_path = f'/path/to/datasets/speedplus_dataset/annotations/instances_{split_name}.json'
IMG_PATH = f"/path/to/datasets/speedplus_dataset/images/{split_name}"
OUTPUT_DIR = Path(f"outputs/grounded_sam2_speedplus_{split_name}_TTA_v4/")


# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)


def tta_none(img):
    return img, ("none", None)

def tta_hflip(img):
    return TF.hflip(img), ("hflip", None)

def tta_vflip(img):
    return TF.vflip(img), ("vflip", None)

def tta_scale(img, scale):
    h, w = img.shape[-2:]
    nh, nw = int(h * scale), int(w * scale)
    return TF.resize(img, (nh, nw)), ("scale", scale)

def tta_color_brightness(img):
    return TF.adjust_brightness(img, 1.1), ("brightness", None)

def tta_color_contrast(img):
    return TF.adjust_contrast(img, 0.9), ("contrast", None)

def tta_color_saturation(img):
    return TF.adjust_saturation(img, 1.05), ("saturation", None)

def tta_color_hue(img):
    return TF.adjust_hue(img, 0.02), ("hue", None)


def inverse_boxes_norm(boxes_norm, ttype, tparam):
    """
    boxes_norm: Nx4 in normalized coords w.r.t. the *augmented* image.
    Returns boxes normalized in the *original* image frame.
    """
    boxes = boxes_norm.clone()
    if ttype == "hflip":
        # x1' = 1 - x2, x2' = 1 - x1
        x1 = 1.0 - boxes[:, 2].clone()
        x2 = 1.0 - boxes[:, 0].clone()
        boxes[:, 0], boxes[:, 2] = x1, x2
    elif ttype == "vflip":
        # vertical flip: invert y coords
        y1 = 1.0 - boxes[:, 3].clone()
        y2 = 1.0 - boxes[:, 1].clone()
        boxes[:, 1], boxes[:, 3] = y1, y2
    # For uniform scale + normalized coords, no change needed.
    return boxes


def merge_wbf(all_boxes, all_scores, all_labels, iou_thr=0.55, skip_box_thr=0.05):
    """
    Inputs are lists over TTA views.
    Returns fused boxes (normalized), scores, labels.
    """
    boxes, scores, labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    return np.array(boxes), np.array(scores), np.array(labels)


# ==== Example usage ====
# model: your trained torchvision detection model
# image: torch tensor [C, H, W], values in [0,1]

tta_cfgs = [
    tta_none,
    tta_vflip,
]

""" V3 TTA CFG 
tta_cfgs = [
    tta_none,
    tta_hflip,
]
"""

""" V2 TTA CFG 
tta_cfgs = [
    tta_none,
    tta_hflip,
    tta_vflip,
]
"""

""" V1 TTA CFG 
tta_cfgs = [
    tta_none,
    tta_hflip,
    lambda im: tta_scale(im, 0.8),
    tta_color_brightness,
    tta_color_contrast,
    tta_color_saturation,
    tta_color_hue,
]
"""

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT

# test_images = os.listdir(IMG_PATH)

with open(gt_json_path) as ff:
    coco_annots = json.load(ff)

all_images = [v['file_name'] for v in coco_annots['images']]

for ind, test_image in enumerate(tqdm(all_images)):

    # if test_image not in all_images:
    #     continue

    # create output directory
    OUTPUT_DIR_IMG = os.path.join(OUTPUT_DIR, test_image)
    os.makedirs(OUTPUT_DIR_IMG, exist_ok=True)
    
    image_path = os.path.join(IMG_PATH, test_image)

    image_source, image = load_image(image_path)

    all_boxes = []
    all_scores = []
    all_labels = []

    for tta in tta_cfgs:
        img_aug, (ttype, tparam) = tta(image)

        with torch.no_grad():
            sam2_predictor.set_image(image_source)

            boxes, confidences, labels = predict(
                model=grounding_model,
                image=img_aug,
                caption=text,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=DEVICE
            )

            # process the box prompt for SAM 2
            h, w, _ = image_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes_torch = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            input_boxes = input_boxes_torch.numpy()

            if len(confidences) == 0:
                continue
        
            # get the top confident box!
            max_ind = int(confidences.argmax())
            input_boxes = input_boxes[max_ind, None]
            confidences = confidences[max_ind, None]
            labels = [labels[max_ind]]

            boxes = torch.from_numpy(input_boxes).cuda() / torch.tensor([w, h, w, h], device=DEVICE)  # normalize
          
            # invert to original coords
            boxes = inverse_boxes_norm(boxes, ttype, tparam)

            all_boxes.append(boxes.cpu().numpy())
            all_scores.append(confidences.cpu().numpy())
            all_labels.append(np.array(list(range(len(labels)))))

    final_boxes, final_scores, final_labels = merge_wbf(all_boxes, all_scores, all_labels)
    # Convert back to pixel coords if you need pixels
    try:
        input_boxes = final_boxes.copy()
        input_boxes[:, [0, 2]] *= w
        input_boxes[:, [1, 3]] *= h
        input_boxes = input_boxes[0, None]
    except:
        input_boxes = all_boxes
    
    class_names = [TEXT_PROMPT]


    if len(final_boxes) == 0:
        # no detection, just create empty json and continue
        input_boxes = []
        scores = 0.0
        # save the results in standard format
        results = {
            "image_path": image_path,
            "annotations" : [
                {
                    "class_name": text,
                    "bbox": input_boxes,
                    "segmentation": [],
                    "score": scores,
                }
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        with open(os.path.join(OUTPUT_DIR_IMG, "grounded_sam2_local_image_demo_results.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        continue

    # FIXME: figure how does this influence the G-DINO model
    # torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = final_scores.tolist()

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(image_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(OUTPUT_DIR_IMG, "groundingdino_annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(OUTPUT_DIR_IMG, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)

    """
    Dump the results in standard format and save as json files
    """

    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": image_path,
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                    "box_score": confidence,
                }
                for class_name, box, mask_rle, score, confidence in zip(class_names, input_boxes, mask_rles, scores, confidences)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        
        with open(os.path.join(OUTPUT_DIR_IMG, "grounded_sam2_local_image_demo_results.json"), "w") as f:
            json.dump(results, f, indent=4)