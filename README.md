# Annotation Free Spacecraft Detection and Segmentation using Vision Language Models

[Samet Hicsonmez](https://scholar.google.com/citations?user=biHfDhUAAAAJ),
[Jose Sosa](https://scholar.google.com/citations?user=R6rtktIAAAAJ),
[Dan Pineau](https://scholar.google.com/citations?user=N55KrR4AAAAJ),
[Inder Pal Singh](https://scholar.google.com/citations?user=EByNsQ0AAAAJ),
[Arunkumar Rathinam](https://scholar.google.com/citations?user=zC2Ri2MAAAAJ),
[Abd El Rahman Shabayek](https://scholar.google.com/citations?user=185kRdEAAAAJ),
[Djamila Aouada](https://scholar.google.com/citations?user=WBmJVSkAAAAJ)

[Interdisciplinary Centre for Security, Reliability, and Trust (SnT), University of Luxembourg](https://www.uni.lu/snt-en/research-groups/cvi2/)

[![arXiv](https://img.shields.io/badge/arXiv-PDF-red)](TODO!!!) 



## Abstract
Vision Language Models (VLMs) have demonstrated remarkable performance in open-world zero-shot visual recognition. However, their potential in space-related applications remains largely unexplored. In the space domain, accurate manual annotation is particularly challenging due to factors such as low visibility, illumination variations, and object blending with planetary backgrounds. Developing methods that
can detect and segment spacecraft and orbital targets without
requiring extensive manual labeling is therefore of critical importance. In this work, we propose an annotation-free detection and segmentation pipeline for space targets using VLMs. Our
approach begins by automatically generating pseudo-labels for
a small subset of unlabeled real data with a pre-trained VLM.
These pseudo-labels are then leveraged in a teacher–student
label distillation framework to train lightweight models. Despite
the inherent noise in the pseudo-labels, the distillation process
leads to substantial performance gains over direct zero-shot VLM inference. Experimental evaluations on the SPARK-2024, SPEED+, and TANGO datasets on segmentation tasks demonstrate consistent improvements in average precision (AP) by up to 10 points. 



## 🖥️ Quick Start

### 1. Clone and Install


First create a new conda environment, and install all required packages.

```
git clone https://github.com/giddyyupp/annotation-free-spacecraft-segmentation
cd annotation-free-spacecraft-segmentation

conda create -n afss python=3.10
conda activate afss
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Then, follow the original Installation instructions present [here](./README_GSAM2.md#installation).


### 2. Datasets
We share the preprocessed json files for both train and val splits in [HF](https://huggingface.co/samethi/annotation-free-spacecraft-segmentation). 
We are not using the synthetic splits of the datasets, except for Tango which is purely synthetic.  

The folder structure should look like below in the end:

```
|-- /path/to/data/dir
    |-- Tango_RoI_and_SS
        |-- annotations
            |-- instances_train_tango.json
            |-- instances_val_tango.json
        |-- images
            |-- train
                |-- images...
            |-- val
                |-- images...
        |-- train_SS_img
        |-- test_SS_img
        |-- train_ROI.json
        |-- test_ROI.json
    
    |-- speedplus_dataset
        |-- annotations
            |-- instances_train_sunlamp_500.json
            |-- instances_val_sunlamp.json
            |-- instances_train_lightbox_500.json
            |-- instances_val_lightbox.json
        |-- images
            |-- train_sunlamp
                |-- images...
            |-- train_lightbox
                |-- images...
            |-- val_lightbox
                |-- images...
            |-- val_sunlamp
                |-- images...
        |-- labels
    
    |-- spark_dataset
        |-- annotations
            |-- instances_train_target_500.json
            |-- instances_val_target_1600.json
        |-- images
            |-- train_target_500
                |-- images...
            |-- val_target_1600
                |-- images...
        |-- labels
```

#### 2.1. Spark
Register and download the dataset from [here](https://cvi2.uni.lu/spark2024/).


#### 2.2. Speed+
Download the Speed+ dataset from this [link](https://purl.stanford.edu/wv398fc4383).


#### 2.3. Tango

Tango dataset is available [here](https://zenodo.org/records/6507864).


## 3. Pipeline


### 3.1. Generating pseudo-annotations

Start with getting zero-shot performance of Grounded SAM2 on space datasets.

Adjust the paths in `grounded_sam2_pseudo_label.py` then run:

```bash
python grounded_sam2_pseudo_label.py
```

### 3.2. Refining using TTA

In the next step, evaluate the performance of Test time augmentation strategies.

Adjust the paths in `grounded_sam2_refine_labels_TTA.py` then run:

```bash
python grounded_sam2_refine_labels_TTA.py
```


### 3.3. Distillation

Now we have refined labels using TTA. Before distillation, we select the predictions with a high confidence. 


Adjust the paths in `prepare_distillation_datasets.py` then run:

```bash
python prepare_distillation_datasets.py
```

You can use the saved json file for the training of [EfficientDet](https://github.com/xuannianz/EfficientDet) and [Yolov11](https://github.com/ultralytics/ultralytics) models.


### 3.4. Inference

For inference, you can use the validation jsons shared in the HF link above.


We share pretrained models using EfficientDet and Yolov11 for both detection and segmentation in [HF](https://huggingface.co/samethi/annotation-free-spacecraft-segmentation).

----------

# Citation

```
@inproceedings{hicsonmez2026afss,
    title={Annotation Free Spacecraft Detection and Segmentation using Vision Language Models}, 
    author={Samet Hicsonmez, Jose Sosa, Dan Pineau, Inder Pal Singh, Arunkumar Rathinam, Abd El Rahman Shabayek, Djamila Aouada},
    year={2026},
    booktitle={IEEE International Conference on Robotics and Automation (ICRA)}
}
```

# Acknowledgement

This codebase is largely built upon:

* [Grounded SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)

We sincerely thank the authors for making their work publicly available.
