# Immune Cell Classification and Detection Project

## Overview
This project aims to create an automated system for identifying and classifying white blood cells (WBCs) in images. It uses a combination of **YOLOv8** for object detection and **Fastai** for classification. The goal is to detect white blood cells in an image and then classify them into five categories:
- **Basophils**
- **Eosinophils**
- **Lymphocytes**
- **Monocytes**
- **Neutrophils**

The model is designed to support research in hematology and immunology and assist in diagnostic workflows.

---

## Dataset

### Structure
The dataset is organized into three main subsets:
- **Train**: Used for model training.
- **Validation**: Used for hyperparameter tuning and performance monitoring during training.
- **Test**: Used to evaluate the final model's performance.

Each subset contains five subfolders corresponding to the cell types:
- `basophil/`
- `eosinophil/`
- `lymphocyte/`
- `monocyte/`
- `neutrophil/`

### Class Distribution
The dataset contains the following number of images for each cell type:
- **Basophils**: 301 images
- **Eosinophils**: 1,066 images
- **Lymphocytes**: 3,609 images
- **Monocytes**: 795 images
- **Neutrophils**: 10,862 images

---

## Model Architecture

### Object Detection (YOLOv8)
- **YOLOv8** is used for object detection to locate white blood cells in images. The YOLOv8 model detects cell locations, and the bounding boxes around detected cells are passed for classification.
- **Training**: The model is fine-tuned on the white blood cell dataset to detect the cells accurately.
- **Classification**: Once cells are detected, a classification model (based on **Fastai** and **ResNet**) is used to classify the cells into one of the five categories mentioned above.

### Features:
- **Data Augmentation**: To address class imbalance and improve model generalization.
- **Loss Function**: Cross-Entropy Loss for multi-class classification.
- **Metrics**: Accuracy, precision, recall, and F1-score are used to evaluate the model.
- **Training Logging**: Training progress and metrics are tracked using **Weights & Biases** (W&B).

---

## Installation

### Setting up the Environment:
Set it up in Colab:

1. Open a new Google Colab notebook.
2. Clone the repository and navigate to the project directory:

```python
!git clone https://github.com/aharoncooper/immune-cell-classification.git
%cd immune-cell-classification
```

3. Install the required dependencies:

```python
!pip install -r requirements.txt
```

4. (Optional) Set up Weights & Biases for logging:

```python
!pip install wandb
!wandb login
```

## Usage
### Data Preparation
Ensure the dataset is organized as follows:

```bash
white_blood_cells/
|-- train/
|   |-- basophil/
|   |-- eosinophil/
|   |-- lymphocyte/
|   |-- monocyte/
|   |-- neutrophil/
|
|-- valid/
|   |-- basophil/
|   |-- eosinophil/
|   |-- lymphocyte/
|   |-- monocyte/
|   |-- neutrophil/
|
|-- test/
    |-- basophil/
    |-- eosinophil/
    |-- lymphocyte/
    |-- monocyte/
    |-- neutrophil/
```

## YOLOv8 Integration
### Object Detection with YOLOv8
The model uses YOLOv8 for detecting white blood cells in images. YOLOv8 performs object detection by drawing bounding boxes around detected cells. After detection, Fastai is used for classifying each detected cell into one of the following categories:

- **Basophils**
- **Eosinophils**
- **Lymphocytes**
- **Monocytes**
- **Neutrophils**
### Steps:
1. Detect Cells: YOLOv8 detects the locations of cells in the image.
2. Classify Detected Cells: After detection, Fastai is used to classify the cells.
3. Visualization: The detected cells are visualized with bounding boxes and the predicted class labels.
### Code in Different Notebooks:
The project is divided into multiple notebooks:

- Notebook 1: Using Fastai to train for classification of immune cells.
- Notebook 2: Testing the classification model.
- Notebook 3: Training the YOLOv8 model for cell detection.
- Notebook 4: Running the final object detection and classification pipeline.
### Steps to Run the Project:
1. Notebook 1: Setup Kaggle API and download the dataset, including renaming and organizing files. Train the model. 
2. Notebook 2: Use the model on test dataset.
3. Notebook 3: Use YOLOv8 for detecting the cells in an image.
4. Notebook 4: Execute the final combined pipeline for detection and classification.
## Results
- Accuracy: The model achieved an accuracy of 85.79% on the test dataset.
- Class-wise Performance: Detailed metrics (precision, recall, F1-score) are available on Weights & Biases (W&B).
- Detection Performance: YOLOv8 is used for cell detection, which outputs bounding boxes around each detected cell for classification.
## Future Work:
Extended Dataset: Adding more images to the dataset, especially for underrepresented classes.

Experimentation: Try different architectures like Vision Transformers or ensemble methods.

Web Interface: Create a web-based interface for real-time white blood cell detection and classification.
## Acknowledgments:
Thanks to the contributors of open-source libraries and datasets that made this project possible, especially the YOLOv8 and Fastai communities.

## License:
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact:
For any questions or collaboration inquiries, please reach out to cooperaharon49@gmail.com.

You can find my model on Hugging Face [here](https://huggingface.co/CooperAharon/white-blood-cell-classifier).
