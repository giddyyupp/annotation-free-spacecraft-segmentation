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

"""
Hyper parameters
"""

# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
TEXT_PROMPT = "spacecraft."

split_name = 'train_sunlamp_500'
gt_json_path = f'/path/to/datasets/speedplus_dataset/annotations/instances_{split_name}.json'
IMG_PATH = f"/path/to/datasets/speedplus_dataset/images/{split_name}"
OUTPUT_DIR = Path(f"outputs/grounded_sam2_speedplus_{split_name}/")


SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DUMP_JSON_RESULTS = True

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

text = TEXT_PROMPT

with open(gt_json_path) as ff:
    coco_annots = json.load(ff)

all_images = [v['file_name'] for v in coco_annots['images']]

for test_image in tqdm(all_images):

    # get the GT box annotation
    idx = all_images.index(test_image)
    gt_bbox = np.array(coco_annots['annotations'][idx]['bbox'], dtype=np.float32) # xywh
    gt_bbox[2] += gt_bbox[0]
    gt_bbox[3] += gt_bbox[1]
    # create output directory
    OUTPUT_DIR_IMG = os.path.join(OUTPUT_DIR, test_image)
    os.makedirs(OUTPUT_DIR_IMG, exist_ok=True)
    
    image_path = os.path.join(IMG_PATH, test_image)

    image_source, image = load_image(image_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # No detection, just save empty dict and continue
    if len(input_boxes) == 0:
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
                    "box_score": scores,
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
    
    max_ind = int(confidences.argmax())
    input_boxes = input_boxes[max_ind, None]
    confidences = confidences[max_ind, None]
    labels = [labels[max_ind]]

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,  # forward only the highest 
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = ['spacecraft']
    class_ids = np.array(list(range(len(class_names))))
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    if confidences[0] < 0.7:
        print(OUTPUT_DIR_IMG)

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

    detections = sv.Detections(
        xyxy=gt_bbox[None, :],  # (n, 4)
        mask=None,  # (n, h, w)
        class_id=np.array(list(range(1)))
    )
    labels_gt = [
        f"spacecraft 1.00"
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_gt)
    cv2.imwrite(os.path.join(OUTPUT_DIR_IMG, "gt_box.jpg"), annotated_frame)

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