# skin-lesion-classification-scratchcnn-tl-gradcam
This project is dedicated to the development and comparison of three deep learning models for binary classification of skin lesions, based on the ISIC 2018-based Binary Classification Dataset (preprocessed and augmented).

The work implements a complete pipeline: from preprocessing "raw" images to training models from scratch and utilizing transfer learning, as well as interpreting their decisions using Grad-CAM.


### Project Goals

1. Build an end-to-end pipeline for dermatoscopic image classification.
2. Train three architectures:
   - Scratch CNN: A custom-built architecture trained from the ground up;
   - MobileNet
   - EfficientNet
3. Compare their performance based on the following metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.
4. Visualize model attention using Grad-CAM and analyze which areas of the image the models focus on.

## 1. Data Preparation

The binary ISIC 2018 dataset was loaded from Hugging Face, containing two balanced classes (0 and 1). The data was split using a stratified approach into training (70%), validation (15%), and testing (15%) subsets.

## 2. Image Preprocessing

An image preprocessing pipeline was implemented using `tf.data`, which includes JPEG decoding, pixel normalization to the [0,1] range, and resizing to 224 x 224. An automatic black-border cropping function was added, and an attempt was made to remove microscope artifacts using morphological operation. 

## 3. Data Augmentation

Data Augmentation was performed using a set of transformations from `tf.keras.layers`: rendom flips, rotations (5%), zoom (10%), as well as minimal adjustments to brightness and contrast (lower than 1%) with subsequent value clipping to the valid range. The pipeline supports caching, parallel processing, and data prefetching to accelerate training.

## 4. Augmentation Visualization

Augmentations were visualized by displaying three pairs of original and augmented images, confirming that the semantics were preserved and transformation were applied correctly.

## 5. Building the Scratch CNN Architecture

Based on the proposed architecture, a convolutional neural network was built from the ground up. The model consists of four convolutional blocks featuring BatchNormalization, ReLU, MaxPooling, and Dropout. It concludes with a GlobalAveragePooling layer and two fully connected (Dense) layers with Dropout for regularization. The output layer contains a single neuron with a sigmoid activation function for binary classification. The model is ready for compilation and training.

```mermaid
graph TD
    subgraph input
        A[Input<br/>224×224×3]
    end
    
    subgraph block1
        B1[Conv2D<br/>32 filters, 3×3] --> B2[BatchNorm] --> B3[ReLU]
        B3 --> C1[Conv2D<br/>32 filters, 3×3] --> C2[BatchNorm] --> C3[ReLU]
        C3 --> D1[MaxPool2D 2×2] --> D2[Dropout 0.25]
    end
    
    subgraph block2
        E1[Conv2D<br/>64 filters, 3×3] --> E2[BatchNorm] --> E3[ReLU]
        E3 --> F1[Conv2D<br/>64 filters, 3×3] --> F2[BatchNorm] --> F3[ReLU]
        F3 --> G1[MaxPool2D 2×2] --> G2[Dropout 0.25]
    end
    
    subgraph block3
        H1[Conv2D<br/>128 filters, 3×3] --> H2[BatchNorm] --> H3[ReLU]
        H3 --> I1[Conv2D<br/>128 filters, 3×3] --> I2[BatchNorm] --> I3[ReLU]
        I3 --> J1[MaxPool2D 2×2] --> J2[Dropout 0.30]
    end
    
    subgraph block4
        K1[Conv2D<br/>256 filters, 3×3] --> K2[BatchNorm] --> K3[ReLU]
        K3 --> L1[MaxPool2D 2×2] --> L2[Dropout 0.35]
    end
    
    subgraph head
        M[GlobalAveragePooling2D] --> N[Dense 128 + ReLU] --> O[Dropout 0.5] --> P[Dense 1 + Sigmoid]
    end
    
    subgraph output
        Q[Binary Output<br/>0/1]
    end
    
    input --> block1
    block1 --> block2
    block2 --> block3
    block3 --> block4
    block4 --> head
    head --> output
```


