import os
import shutil
import json


CONFIDENCE_THRESHOLD = 0.60  # TODO: adjustable

dataset = 'speedplus'
split_names = ['train_sunlamp_500'] # TODO: update for ligthbox

for split_name in split_names:
    pred_json_file_coco_train = f'/path/to/datasets/speedplus_dataset/annotations/instances_{split_name}_gsam2_tta_v4.json'
    pred_json_file_coco_train_conf_removed = f"/path/to/datasets/speedplus_dataset/annotations/instances_{split_name}_gsam2_tta_v4_conf_filtered_060.json"
  
    with open(pred_json_file_coco_train, "r") as f:
        coco_data = json.load(f)

    annots = coco_data['annotations']
    images = coco_data['images']
    images_keep= []
    annotations_keep = [] 

    for annot in annots:
        if annot['score'] >= CONFIDENCE_THRESHOLD:
            annotations_keep.append(annot)
            images_keep.append(images[annot['image_id']])

    coco_data['annotations'] = annotations_keep
    coco_data['images'] = images_keep

    with open(pred_json_file_coco_train_conf_removed, 'w') as ff:
        json.dump(coco_data, ff)

    print(f'Number of training images remain:{len(images_keep)}')
