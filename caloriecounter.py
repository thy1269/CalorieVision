# food_calorie_yolo_app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import os

# Sample calorie database
calorie_database = {
    "bread pudding": 400,
    "steak": 700,
    "ramen": 500
}

# Load pre-trained YOLO model
@st.cache_resource
def load_yolo_model():
    model_path = "runs/train/custom_food_model/weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        st.write(f"Loaded pre-trained model from: {model_path}")
        st.write("Model class names:", model.names)
    else:
        st.error(f"Custom model not found at {model_path}. Please train the model first.")
        raise FileNotFoundError(f"Model file {model_path} not found.")
    return model

# Detect food using YOLO
def detect_food(image, model, conf_threshold=0.25):
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img_cv, conf=conf_threshold)
    
    detected_foods = []
    st.write("Raw detections (class_id, x_center, y_center, width, height, confidence):", results[0].boxes.data.tolist())
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls)
            label = model.names[class_id]
            confidence = float(box.conf)
            food_name = label.lower()
            st.write(f"Detected: {food_name} (Class ID: {class_id}, Confidence: {confidence:.2f})")
            detected_foods.append((food_name, confidence))
    
    return detected_foods, results

# Main Streamlit app
def main():
    st.title("Food Calorie Scanner with YOLO")
    st.write("Upload an image of food to detect items and estimate calories using a pre-trained YOLO model!")

    # Load pre-trained model
    try:
        model = load_yolo_model()
    except FileNotFoundError:
        return

    # Confidence threshold slider (lowered default)
    conf_threshold = st.slider("Confidence Threshold", 0.05, 1.0, 0.1, 0.05)

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect food and estimate calories
        with st.spinner("Detecting food items..."):
            detected_foods, results = detect_food(image, model, conf_threshold)
            time.sleep(1)

        # Show detection results
        st.subheader("Detected Food Items and Calories")
        total_calories = 0
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detected Items", use_column_width=True)

        if detected_foods:
            for food_name, confidence in detected_foods:
                if food_name in calorie_database:
                    calories = calorie_database[food_name]
                    total_calories += calories
                    st.write(f"- {food_name.capitalize()} (Confidence: {confidence:.2f}): {calories} calories")
                else:
                    st.write(f"- {food_name.capitalize()} (Confidence: {confidence:.2f}): Calorie data not available")
            st.success(f"Total Estimated Calories: {total_calories} kcal")
        else:
            st.warning("No food items detected.")

        # Option to download annotated image
        annotated_image_pil = Image.fromarray(annotated_image)
        st.download_button(
            label="Download Annotated Image",
            data=annotated_image_pil.tobytes(),
            file_name="annotated_food.jpg",
            mime="image/jpeg"
        )

if __name__ == "__main__":
    main()