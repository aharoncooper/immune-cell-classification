import os
import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt

from detect_and_classify import (
    load_models,
    detect_and_classify,
    visualize_detections
)


def main():
    st.title('White Blood Cell Classifier')
    st.write('Upload an image to detect and classify white blood cells')

    # Define paths with full absolute path
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Paths to models and data
    yolo_model_path = os.path.join(base_path, 'models', 'yolov8_best.pt')
    fastai_model_path = os.path.join(base_path, 'models', 'best_model.pth.pth')
    data_path = os.path.join(base_path, 'data', 'white blood cells')

    # Validate paths
    if not os.path.exists(yolo_model_path):
        st.error(f"YOLO model not found at {yolo_model_path}")
        return

    if not os.path.exists(fastai_model_path):
        st.error(f"FastAI model not found at {fastai_model_path}")
        return

    if not os.path.exists(data_path):
        st.error(f"Data path not found at {data_path}")
        return

    # Load models (only once)
    try:
        if 'yolo_model' not in st.session_state:
            st.session_state.yolo_model, st.session_state.fastai_model, st.session_state.dls = load_models(
                yolo_model_path,
                fastai_model_path,
                data_path
            )
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Save uploaded file temporarily
        temp_image_path = 'temp_uploaded_image.png'
        image.save(temp_image_path)

        try:
            # Detect and classify
            detected_image, cell_classifications = detect_and_classify(
                temp_image_path,
                st.session_state.yolo_model,
                st.session_state.fastai_model
            )

            # Visualize detections
            fig = visualize_detections(detected_image, cell_classifications)

            # Display results
            st.pyplot(fig)

            # Display classification details
            st.subheader('Cell Classifications')
            for cell in cell_classifications:
                st.write(
                    f"Cell: {cell['class']} "
                    f"(Confidence: {cell['confidence']:.2f})"
                )

        except Exception as e:
            st.error(f"Error in detection and classification: {e}")


if __name__ == '__main__':
    main()