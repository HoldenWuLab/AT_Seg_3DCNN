# ACD3DUNet
Attention-based competitive dense (ACD) 3D U-Net with a novel frequency balancing Dice loss to segment subcutaneous and visceral adipose tissue (SAT/VAT) !

This page shared the codes for the manuscript submitted to Journal of Magnetic Resonance in Medicine under the title "Evaluation of Automated Abdominal Adipose Tissue Segmentation and Volume Quantification using 3D Convolutional Neural Networks with Multi-Contrast MRI Inputs"

## Overview
Subcutaneous and visceral adipose tissue (SAT/VAT) are potential biomarkers of risk for metabolic diseases. Manual segmentation of SAT/VAT on MRI is the reference standard but is challenging to deploy. Emerging neural networks for automated segmentation still have suboptimal VAT segmentation performance. A recently proposed attention-based competitive dense (ACD) 3D U-Net leverages full field-of-view volumetric multi-contrast MRI inputs (opposed phase echo image, water image, and fat image) and a novel loss function, requency balancing dice loss (FBDL), designed for VAT.

### Inputs: 
Full-volume multi-contrast inputs. Opposed phase echo image (TE_{OP}), W and F images should be stacked together channel dimension.

### Outputs: 
(1) SAT mask
(2) VAT mask
(3) Background mask

## Usage
ACD 3D U-Net is a segmentation network specifically desogned for abdominal SAT and VAT segmentation. There are some initial and simple pre-processing steps that are performed once the input h5 files are read and converted to numpy arrays. 

ACD 3D U-Net was implemented using PyTorch. \
(1) "main.py" contains the script for training ACD 3D U-Net \
(2) 3D U-Net and ACD 3D U-NET can be found in "models_all.py" \
(3) Loss functions can be found in the folder "losses.py" \
(4) "ATSeg_load_HAT_3D.py" includes the pre-processing and the loading of the 3D data from h5 files


Loss function choice: \
2 different loss functions are available in this repository \
(1) 3D Standard Weighted Dice Loss (WDL) \
(2) 3D Frequency Balancing Boudnary Emphasizing Dice Loss (FBDL)
Different parameter settings are required for WDL and FBDL. The example code in main.py here uses FBDL. 
