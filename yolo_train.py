# This file is for YOLO training
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


# Use the model
if __name__ == '__main__':
    results = model.train(data="config.yaml", batch=16, momentum=0.937,
                          epochs=175, lr0=0.01,verbose=False)  # train the model 