import os
import torch
from ultralytics import YOLO
from fastai.vision.all import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def load_models(yolo_model_path, fastai_model_path, data_path):
    """
    Load YOLO and FastAI models with proper DataLoaders
    """
    # Validate model and data paths
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"YOLO model not found at {yolo_model_path}")

    if not os.path.exists(fastai_model_path):
        raise FileNotFoundError(f"FastAI model not found at {fastai_model_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found at {data_path}")

    # Load YOLO model
    try:
        yolo_model = YOLO(yolo_model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise

    # Create DataLoaders with the correct data path
    dls = ImageDataLoaders.from_folder(
        data_path,  # Use the data path with Train, Valid, Test folders
        valid='Valid',  # Specify validation folder
        test='Test',  # Specify test folder
        train='Train',  # Specify train folder
        item_tfms=Resize(224),
        bs=32
    )

    # Load FastAI model
    try:
        model = vision_learner(dls, resnet18, metrics=[accuracy])

        # Use torch.load with explicit map_location
        state_dict = torch.load(fastai_model_path, map_location=torch.device('cpu'))

        # Load state dict with strict=False to handle potential minor mismatches
        model.load_state_dict(state_dict, strict=False)

        # Set the model to evaluation mode
        model.eval()
    except Exception as e:
        print(f"Error loading FastAI model: {e}")
        raise

    return yolo_model, model, dls


def classify_cells(img, model):
    """
    Classify a cell image using FastAI model
    """
    pred_class, pred_idx, outputs = model.predict(img)
    return pred_class, outputs


def detect_and_classify(image_path, yolo_model, fastai_model):
    """
    Detect cells using YOLO and classify each cell with FastAI
    """
    # Load the image
    img = Image.open(image_path)

    # Run YOLO object detection
    results = yolo_model(image_path)

    # Extract YOLO results
    boxes = results[0].boxes.xyxy  # Bounding box coordinates
    labels = results[0].boxes.cls  # Predicted labels

    # Prepare to store detection results
    cell_classifications = []

    # Loop through each detected cell and classify it
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = map(int, [round(x1), round(y1), round(x2), round(y2)])

        # Crop the detected cell region
        cell_img = img.crop((x1, y1, x2, y2))

        # Classify the cropped cell
        pred_class, outputs = classify_cells(cell_img, fastai_model)

        cell_classifications.append({
            'bbox': [x1, y1, x2, y2],
            'class': str(pred_class),
            'confidence': float(outputs.max())
        })

    return img, cell_classifications


def visualize_detections(img, cell_classifications):
    """
    Create a visualization of cell detections
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    for cell in cell_classifications:
        x1, y1, x2, y2 = cell['bbox']

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        # Add text label with class and confidence
        ax.text(
            x1, y1 - 10,
            f'{cell["class"]}: {cell["confidence"]:.2f}',
            color='red', fontsize=12
        )

    plt.axis('off')
    plt.tight_layout()
    return fig