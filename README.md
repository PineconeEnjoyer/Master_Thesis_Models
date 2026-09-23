# Automatic Assessment of Skin Lesions using Deep Learning 🩺

Welcome to my Master's Thesis Models GitHub repository! This project contains the deep learning models and data preprocessing pipelines designed for a web-based application that automatically assesses skin lesions based on user-uploaded images. The system aims to recognize and classify common skin changes, such as moles, discolorations, and melanoma symptoms, to support early-stage diagnostic decisions.

---

## Project Overview

This repository represents the analytical module of a Master's Thesis developed at the Silesian University of Technology. The primary objective is to compare different neural network architectures in terms of their classification performance on dermatological data. To achieve high accuracy, the project implements a robust preprocessing pipeline to handle imbalanced datasets and artifacts (like hair) before training four distinct architectures: a custom Convolutional Neural Network (CNN), ResNet, EfficientNet, and Vision Transformer.

---

## Repository Structure

The project code is modularized into scripts handling data preparation, image preprocessing, and specific model architectures. 

| File Name | Description |
| :--- | :--- |
| **`HairRemoval.py`** | An OpenCV-based script that isolates and removes hair from the skin lesion images using morphological operations (Blackhat), thresholding, and Telea inpainting. |
| **`Preprocessing.py`** | Handles dataset splitting (stratified group splits to prevent data leakage), data augmentation (flips, affine transformations, color jitter), image cropping around the segmented mask, and calculates class weights for the `WeightedRandomSampler`. |
| **`CNN.py`** | Implementation of a custom Convolutional Neural Network architecture with batch normalization, max pooling, and dropout layers. |
| **`ResNet.py`** | A script utilizing a pre-trained ResNet50 model. It freezes the base layers while unfreezing `layer3` and `layer4` for fine-tuning, and implements a custom classifier with dropout. |
| **`EfficientNet.py`** | Implements a pre-trained EfficientNet-B3 model, unfreezing the last two blocks for fine-tuning, and utilizes Mixed Precision to optimize memory and speed. |
| **`Thesis_Document`** | The official project declaration detailing the scope, goals, and academic requirements of the Master's Thesis. |

---

## Built With

* **Python** - The primary programming language used for the entire analytical module.
* **PyTorch** - The core deep learning framework used for building, training, and evaluating the models.
* **OpenCV** - Utilized for advanced image preprocessing, specifically for contour detection and inpainting to remove hair from lesion images.
* **Scikit-Learn** - Applied for stratified dataset splitting, computing class weights, and generating classification reports and confusion matrices.
* **Matplotlib and Seaborn** - Used to visualize training history (accuracy and loss curves) and confusion matrices.

---

## How It Works

1. **Data Preprocessing:** The raw HAM10000 dataset images are first passed through the hair removal algorithm, which isolates hair pixels using median blur and Blackhat morphology, then fills the gaps using OpenCV inpainting. 
2. **Dataset Preparation:** The data is split into training, validation, and testing sets using a stratified approach based on `lesion_id` to ensure no data leakage occurs. Images are cropped around the lesion using provided segmentation masks and augmented to increase dataset variety.
3. **Model Training:** Neural networks are trained using the prepared DataLoaders. The training loops utilize Learning Rate Schedulers, Early Stopping, and class weighting to handle imbalanced data effectively.
4. **Evaluation:** After training, each model is evaluated on the test set. The system automatically generates and saves a classification report, a confusion matrix heatmap, and a plot of the training/validation accuracy and loss.

---

## Acknowledgments

This project was developed as a Master's Thesis at the Department of Cybernetics, Nanotechnology and Data Processing, Silesian University of Technology. Thank you for taking the time to explore this repository and review the deep learning models designed for dermatological diagnostics! 🎓
