# Custom YOLOv11 Model Training for Food Detection

This script trains a YOLOv11 model on a custom dataset for food detection using the Ultralytics YOLO library. It initializes a pre-trained YOLOv11 model, trains it on your dataset, and saves the trained weights for future use.

## Prerequisites

To run this script, ensure you have the following:

- **Ultralytics YOLO library**: Install it using pip:
  ```bash
  pip install ultralytics

## Usage

Follow these steps to train your model:

1. **Review Parameters**: The script includes configurable parameters at the top:
   - `DATA_YAML = "data.yaml"`: Path to your dataset configuration file.
   - `EPOCHS = 10`: Number of training epochs (one full pass through the dataset).
   - `IMGSZ = 640`: Image size for training (resized to 640x640 pixels).
   - `MODEL = "yolo11m.pt"`: Pre-trained YOLOv11 model to use as a starting point (medium variant).

   Modify these in the script if needed.

2. **Run the Script**: Execute the script from the command line:
   ```bash
   python train_yolo_model.py

3. **Monitor Progress**: Training progress (e.g., loss, mAP) will be printed to the console. The process may take time depending on your dataset size and hardware.

4. **Output**: After training, the model weights are saved in:
```bash
runs/train/custom_food_model/weights/




