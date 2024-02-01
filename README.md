# Automated Abdominal Adipose Tissue Segmentation and Volume Quantification on Longitudinal MRI using 3D Convolutional Neural Networks with Multi-Contrast Inputs
This page shared the codes for the paper published in  Magnetic Resonance Materials in Physics, Biology and Medicine under the title "Automated Abdominal Adipose Tissue Segmentation and Volume Quantification on Longitudinal MRI using 3D Convolutional Neural Networks with Multi-Contrast Inputs"

Please follow this link to access the paper:
https://link.springer.com/article/10.1007/s10334-023-01146-3?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20240201&utm_content=10.1007/s10334-023-01146-3 

Please cite this work if you use the code or the neural network weights: 
Kafali, S.G., Shih, SF., Li, X. et al. Automated abdominal adipose tissue segmentation and volume quantification on longitudinal MRI using 3D convolutional neural networks with multi-contrast inputs. Magn Reson Mater Phy (2024). https://doi.org/10.1007/s10334-023-01146-3

# 3D nnU-Net
3D nnU-Net framework with weighted Dice loss (WDL) to segment subcutaneous and visceral adipose tissue (SAT/VAT)

# ACD 3D UNet
Attention-based competitive dense (ACD) 3D U-Net with a novel frequency balancing Dice loss to segment subcutaneous and visceral adipose tissue (SAT/VAT)

## Overview
Subcutaneous and visceral adipose tissue (SAT/VAT) are potential biomarkers of risk for metabolic diseases. Manual segmentation of SAT/VAT on MRI is the reference standard but is challenging to deploy. Emerging neural networks for automated segmentation still have suboptimal VAT segmentation performance. 
Here, we present automated abdominal SAT/VAT segmentation on longitudinal MRI in adults with overweight/obesity using attention-based competitive dense (ACD) 3D U-Net and 3D nnU-Net with full field-of-view volumetric multi-contrast inputs. 

### Inputs: 
Full-volume multi-contrast inputs. Opposed phase echo image (TE_{OP}), W and F images should be stacked together channel dimension. This code uses pre-saved h5 files that include all the images per each subject in the corresponding training/validation/test dataset, then convert them to numpy arrays. 

### Outputs: 
(1) SAT mask \
(2) VAT mask \
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

Funding: \
Research reported in this publication was supported by the National Institute Of Diabetes And Digestive And Kidney Diseases of the National Institutes of Health under Award Number R01DK124417. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

Weights: \
Links will be provided later. Please contact the developers for more information; skafali@mednet.ucla.edu 

Licensing: \
Code: The code for the 3D convolutional neural networks (CNNs) is available at https://github.com/HoldenWuLab/AT_Seg_3DCNN under an Academic Software License: © 2023 UCLA (“Institution”).

Neural network weights: The final weights for the trained 3D CNNs can be used under a non-commercial Creative Commons (Attribution-NonCommercial-NoDerivatives 4.0 International) license. <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />The final neural network weights of this work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
For more information, please visit: https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode


