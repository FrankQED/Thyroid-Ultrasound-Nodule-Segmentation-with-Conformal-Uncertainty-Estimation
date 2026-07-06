# Thyroid Nodule Segmentation in Ultrasound Images with Conformal Uncertainty Estimation
Code for "Thyroid Nodule Segmentation in Ultrasound Images with Conformal Uncertainty Estimation" from CIBB2K26

## Introduction
Thyroid cancer is the most common endocrine malignancy, accounting for about 3.4% of newly diagnosed cancers worldwide, and approximately 10% of all thyroid nodules are malignant.
The primary diagnostic modality is ultrasound; however, image interpretation involves a subjective component that introduces both inter- and intra-observer variability.
Computer-Aided Diagnosis (CAD) systems can reduce this variability and provide clinicians with a second opinion. 
The first stage of any thyroid cancer CAD pipeline is nodule segmentation: an accurate mask is the starting point for downstream malignancy classification and metastatic-risk quantification.

In this work, we propose a reliable nodule segmentation baseline.
We adopt DeepLabV3+~\cite{deeplab} with an EfficientNet-B5 backbone pre-trained on ImageNet, and we explore the performance of three channel-attention modules: Squeeze-and-Excitation (SE), Convolutional Block Attention Module (CBAM) and Grouped Channel Attention (GCA); combined with two loss functions.
The GCA module has previously been shown to be effective in a multimodal CLIP-based framework for thyroid segmentation, but here we evaluate it in a purely visual setting.
Finally, we add a conformal prediction layer to the trained model that produces pixel-level uncertainty maps with coverage guarantees.
