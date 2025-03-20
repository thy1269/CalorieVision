# train_yolo_model.py
from ultralytics import YOLO

# Training parameters
DATA_YAML = "data.yaml"  # Path tmo your data.yaml file
EPOCHS = 10              # Number of training epochs
IMGSZ = 640              # Image size for training
MODEL = "yolo11m.pt"     # Base model (nano version)

def train_model():
    # Load the base model
    model = YOLO(MODEL)
    
    # Train the model
    print("Starting training...")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        project="runs/train",
        name="custom_food_model",
        exist_ok=True
    )
    print("Training complete! Weights saved in runs/train/custom_food_model/weights/")

if __name__ == "__main__":
    train_model()