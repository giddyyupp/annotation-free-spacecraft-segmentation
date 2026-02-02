import os
import numpy as np
import json
from tqdm import tqdm

from collections import defaultdict

try:
    from pycocotools import mask as maskUtils
except ImportError as e:
    raise ImportError(
        "pycocotools is required for COCO-style RLE decoding and IoU. "
        "Install with: pip install pycocotools"
    ) from e

from PIL import Image

def rle_for_pycoco(rle):
    """
    Ensure RLE is in a format pycocotools likes:
    - counts: bytes
    - size: [H, W]
    Returns a new dict (does not mutate input).
    """
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")  # or 'ascii'
    return {"size": list(rle["size"]), "counts": counts}


def load_binary_mask(path):
    """
    Load a binary mask GT image from disk.
    Assumes mask pixels are 0 (background) and >0 (foreground).
    Returns: H x W numpy array of dtype=bool
    """
    img = Image.open(path).convert("L")  # ensure grayscale
    mask = np.array(img)
    return mask > 0  # convert to boolean mask

def load_binary_mask_tango(path):
    """
    Load a binary mask GT image from disk.
    Assumes mask pixels are 0 (background) and >0 (foreground).
    Returns: H x W numpy array of dtype=bool

    Concerning the Semantic Segmentation Labels, they are provided as RGB masks named as "filename_mask.png" where "filename" is the filename of the image of the training set or the test set to which a specific mask is referred. The RGB images are such that the R channel corresponds to the spacecraft, the G channel corresponds to the Earth (if present), and the B channel corresponds to the background (deep space). Per each channel the pixels have non-zero value only in correspondence of the object that they represent (Tango, Earth, Deep Space). 

    """
    img = Image.open(path).convert("RGB")  # ensure grayscale
    mask = np.array(img)
    return mask[:, :, 0] > 0  # spacecraft channel.

def encode_binary_mask_to_rle(mask_bool):
    """
    Convert a binary mask (H x W, bool or {0,1}) to COCO RLE dict.
    Ensures Fortran order and uint8 type as pycocotools expects.
    """
    if mask_bool.dtype != np.uint8:
        mask_bool = mask_bool.astype(np.uint8)
    rle = maskUtils.encode(np.asfortranarray(mask_bool))
    # pycocotools returns counts as bytes; convert to str for JSON-friendliness
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def coco_like_segmentation_ap(predictions, gt_masks_by_image, iou_thresholds=None):
    """
    Compute COCO-like AP for instance segmentation for a single class.

    Args:
        predictions: list of dicts, each:
            {
              "image_id": <hashable>,
              "rle": <COCO RLE dict OR binary mask np.ndarray>,
              "score": float
            }
            - If you pass a binary mask instead of RLE, it will be encoded on-the-fly.

        gt_masks_by_image: dict mapping image_id -> list of GT instance masks.
            Each GT mask can be:
              - binary mask np.ndarray (H x W)
              - COCO RLE dict
            If you have a single binary mask image with multiple instances merged,
            you should first split it into instance masks (connected components).

        iou_thresholds: iterable of IoU thresholds. Default = np.arange(0.50, 0.96, 0.05)

    Returns:
        metrics: dict with
          {
            "AP": mean AP over IoU thresholds,
            "AP50": AP at 0.50,
            "AP75": AP at 0.75,
            "AP_per_threshold": {thr: AP_thr, ...},
          }
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.50, 0.96, 0.05)

    # --- Normalize GTs to RLE per image ---
    gt_rles_by_image = {}
    for img_id, masks in gt_masks_by_image.items():
        rles = []
        for m in masks:
            if isinstance(m, dict) and "counts" in m and "size" in m:
                rles.append(m)
            else:
                # Assume binary mask
                rles.append(encode_binary_mask_to_rle(m > 0))
        gt_rles_by_image[img_id] = rles

    # --- Normalize predictions to RLE and gather globally ---
    preds = []
    for p in predictions:
        img_id = p["image_id"]
        score = float(p["score"])
        rle = p["rle"]
        if not (isinstance(rle, dict) and "counts" in rle and "size" in rle):
            # Assume binary mask array
            rle = encode_binary_mask_to_rle(rle > 0)
        preds.append({"image_id": img_id, "rle": rle, "score": score})

    # Sort predictions by confidence (desc)
    preds.sort(key=lambda x: x["score"], reverse=True)

    # Precompute GT "crowd" flags if you need them; here assume all non-crowd
    # (COCO handles crowd differently, but for typical use we set iscrowd=0).
    iscrowd_by_image = {img_id: [0] * len(rles) for img_id, rles in gt_rles_by_image.items()}

    # For each IoU threshold, do matching and compute AP
    ap_per_thr = {}

    for thr in iou_thresholds:
        # Track which GTs are already matched per image
        gt_matched = {img_id: np.zeros(len(gt_rles), dtype=bool)
                      for img_id, gt_rles in gt_rles_by_image.items()}

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        # Process predictions in order
        for i, pred in enumerate(preds):
            img_id = pred["image_id"]
            pred_rle = pred["rle"]

            gt_rles = gt_rles_by_image.get(img_id, [])
            if len(gt_rles) == 0:
                # No GT in this image -> false positive
                fp[i] = 1.0
                continue

            iscrowd = iscrowd_by_image[img_id]
            # Compute IoUs between this prediction and all GTs in the same image
            # maskUtils.iou expects lists of RLEs; returns [len(preds) x len(gts)]
            ious = maskUtils.iou([pred_rle], gt_rles, iscrowd)[0]  # shape: (num_gt,)

            # Find best unmatched GT above threshold
            best_gt = -1
            best_iou = thr
            for j, iou in enumerate(ious):
                if iou >= best_iou and not gt_matched[img_id][j]:
                    best_iou = iou
                    best_gt = j

            if best_gt >= 0:
                tp[i] = 1.0
                gt_matched[img_id][best_gt] = True
            else:
                fp[i] = 1.0

        # Precision-Recall
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / (sum(len(v) for v in gt_rles_by_image.values()) + 1e-12)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        # 101-point interpolated precision (COCO)
        # For each r in {0.00, 0.01, ..., 1.00}, precision_interp[r] = max_{recall >= r} precision
        recall_points = np.linspace(0, 1, 101)
        precision_interp = np.zeros_like(recall_points)
        for k, r in enumerate(recall_points):
            mask = recalls >= r
            precision_interp[k] = np.max(precisions[mask]) if np.any(mask) else 0.0

        ap_per_thr[thr] = float(np.mean(precision_interp))

    metrics = {
        "AP": float(np.mean(list(ap_per_thr.values()))),
        "AP50": float(ap_per_thr.get(0.50, np.nan)),
        "AP75": float(ap_per_thr.get(0.75, np.nan)),
        "AP_per_threshold": {float(k): float(v) for k, v in ap_per_thr.items()},
    }
    return metrics

# Example inputs (single class):
# predictions: list of dicts with COCO RLEs and scores
# gt_masks_by_image: dict image_id -> list of binary masks (one per GT instance)

# Suppose you have for each image a list of (H x W) boolean arrays for GTs:
# gt_masks_by_image = {
#    "img_001": [gt_mask1_bool, gt_mask2_bool],
#    "img_002": [gt_mask1_bool],
#    ...
# }

# And predictions are COCO RLEs and scores:
# predictions = [
#   {"image_id": "img_001", "rle": pred_rle_0, "score": 0.93},
#   {"image_id": "img_001", "rle": pred_rle_1, "score": 0.77},
#   {"image_id": "img_002", "rle": pred_rle_2, "score": 0.88},
#   ...
# ]


if __name__ == "__main__":
    method = 'grounded_sam2'
    dataset = 'speedplus'
    split = 'sunlamp'
    prediction_dir = f'./outputs/grounded_sam2_{dataset}_val_{split}_TTA_v4'

    gt_dir = f'/path/to/dataset/speedplus/masks/{split}/masks'
    # gt_dir = f'/path/to/dataset/spark2024/annotations_coco/{split}/masks'
    # gt_dir = f'/path/to/dataset/tango_dataset/Tango_RoI_and_SS/test_SS_img'

    pred_images = os.listdir(prediction_dir)
    predictions = []

    print(prediction_dir)

    for pred_image in tqdm(pred_images):
        try:
            # with open(os.path.join(prediction_dir, pred_image, f'{method}_local_image_demo_results.json'), 'r') as ff:
            with open(os.path.join(prediction_dir, pred_image, f'grounded_sam2_local_image_demo_results.json'), 'r') as ff:
                predictions_data = json.load(ff)
            pred_seg = predictions_data["annotations"][0]["segmentation"]
            pred_rle_json = {"size": pred_seg['size'], "counts": pred_seg['counts']}  # counts is str
            pred_rle_for_eval = rle_for_pycoco(pred_rle_json)       # counts -> bytes
            pred_dict = {"image_id": pred_image, "rle": pred_rle_for_eval, "score": predictions_data["annotations"][0]["score"]}
        except:
            continue

        predictions.append(pred_dict)

    gt_masks_by_image = {}

    for pred_image in pred_images:
        if dataset == 'tango':
            gt_mask = [load_binary_mask_tango(os.path.join(gt_dir, pred_image.split('.')[0] + '_mask.' + pred_image.split('.')[1]))]
        else:
            gt_mask = [load_binary_mask(os.path.join(gt_dir, pred_image))]
        gt_masks_by_image[pred_image] = gt_mask

    metrics = coco_like_segmentation_ap(predictions, gt_masks_by_image)
    print(metrics)
