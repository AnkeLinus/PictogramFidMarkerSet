# PictogramFiducialMarkerSet
This repository is used to train a YOLOv12n model on a dataset. Follow the steps below to train it yourself. Additionally, an evaluation file is provided to gain insights in the performance of the model and dataset. This repository was used in the publication cited below. 

With this code a dataset was tested by training and evaluating the provided YOLOv12n model. The dataset contained 1008 images (without augmentation), picturing random placement of 8 pictograms in domestic home and workplace environment. These pictograms show 8 different daily tasks. The aim of using this object detection approach was to use these markers as interactable objects in a eye-tracking control for the user and further use them as fiducial markers for robotic arm control.  


## Getting started

1. Clone the git into your Filesystem.

2. Create an environment and install all requirements. The code was developed and trained on Python 3.11.9. Install requirements command:

    ```pip install -r requirements.txt```

3. For Training: 
    1. Download your favored Yolo Model 
    2. In Training.py: Go to Roboflow: https://app.roboflow.com/wlrisemanticlables/wlri-semantic_labels/8. Click Download Dataset -> Show Download Code and continue. Enter the API-Key in line 12 of Training.py. Check if Lines 13 to 15 fit with the generated code. 


4. For Evaluation:
    1. In the git a pretrained model on the pictogram dataset is available and integrated. You can use this model to test the evaluation code. 
    2. Change the following parameters in Evaluation.py: In lines 125 to 127 change PATH_TO_GIT into your file directory to the pictodataset folder. 

5. Either train the model or run evaluation to recreate outcomes. In both cases make sure, that the dataset import (lines 12 to 15 in Training.py) is uncommented in the first run or import the dataset manually. 


## Acknowledgement
This repository contains code from following authors and websites: 

- Roboflow - Dwyer, B., Nelson, J., Hansen, T., et al. (2025). Roboflow (Version 1.0) [Software]. Available from https://roboflow.com. Computer vision.
- YOLOv12 - Yunjie Tian and Qixiang Ye and David Doermann (2025). YOLOv12: Attention-Centric Real-Time Object Detectors. arXiv: 2502.12524 cs.CV, https://arxiv.org/abs/2502.12524.
- Katrin-Misel Ponomarjova - https://github.com/katrinmisel/assistive_training/blob/main/training.ipynb

## License and Citation
This git is free to use for everyone. Please, when used in academic context cite the following Paper:

**Anonymized Authors. Robust and Scalable Task Selection for Humans and Robots with the Use of Pictograms as Fiducial Markers, in Added after Review (2025).**

Bibtex:
```
authors = 'Anonymized',
title = 'Robust and Scalable Task Selection for Humans and Robots with the Use of Pictograms as Fiducial Markers',
journal = '', 
volume = '',
issue = '',
year = ''
```
