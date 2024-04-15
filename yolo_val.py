# This file is for YOLO varification
from ultralytics import YOLO

# Load a model
model = YOLO('models/175.pt')

# Use the model
if __name__ == '__main__':
    metrics = model.val(data='config.yaml', batch=16)  # evaluate model performance on the validation set