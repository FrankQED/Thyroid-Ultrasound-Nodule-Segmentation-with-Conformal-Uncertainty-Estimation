# Thyroid Nodule Segmentation in Ultrasound Images with Conformal Uncertainty Estimation
Code for "Thyroid Nodule Segmentation in Ultrasound Images with Conformal Uncertainty Estimation" from CIBB2K26

## Introduction
Thyroid cancer is the most common endocrine malignancy, accounting for about $3.4\%$ of newly diagnosed cancers worldwide, and approximately $10\%$ of all thyroid nodules are malignant.
The primary diagnostic modality is ultrasound; however, image interpretation involves a subjective component that introduces both inter- and intra-observer variability.
Computer-Aided Diagnosis (CAD) systems can reduce this variability and provide clinicians with a second opinion. 
The first stage of any thyroid cancer CAD pipeline is nodule segmentation: an accurate mask is the starting point for downstream malignancy classification and metastatic-risk quantification.

In this work, we propose a reliable nodule segmentation baseline.
We adopt DeepLabV3+~\cite{deeplab} with an EfficientNet-B5~\cite{efficientnet} backbone pre-trained on ImageNet, and we explore the performance of three channel-attention modules: Squeeze-and-Excitation (SE)~\cite{se}, Convolutional Block Attention Module (CBAM)~\cite{cbam} and Grouped Channel Attention (GCA)~\cite{gcaseg}; combined with two loss functions.
The GCA module has previously been shown to be effective in a multimodal CLIP-based framework for thyroid segmentation~\cite{gcaseg}, but here we evaluate it in a purely visual setting.
Finally, we add a conformal prediction layer~\cite{cp} to the trained model that 
