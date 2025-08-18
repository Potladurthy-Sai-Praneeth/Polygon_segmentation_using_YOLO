import os

# Train configurations
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MODEL_SIZE = 's'  # YOLO model size ('n', 's', 'm', 'l', 'x')
IMGSZ = 640  


# Dataset configurations
DATASET_PATH = 'data/dataset.parquet'
DATASET_YAML = 'outputs/dataset.yaml'  # Path to dataset YAML configuration


# Output paths
MODEL_PATH = "outputs/yolo_training/weights/best.pt"
IMAGES_BASE_PATH = "data/images"       
OUTPUT_DIR = "outputs"
