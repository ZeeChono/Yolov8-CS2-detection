# Yolov8-CS2-detection
This project is for self-study about applying the Yolo object detection algorithm in real-time gaming CS-2.

## Table of Contents
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Getting Started](#getting-started)
  - [Usage](#usage)

## Features

- Detecting the two classes of characters(T, CT) in the game
  image
- Detecting the head positions(Th, CTh) from two type of classes
  image

## Getting Started


This project
Ensure you have the following dependencies installed before running the project:

- [Python](https://www.python.org/downloads/) (version >= 3.x)
- [PyTorch](https://pytorch.org/get-started/locally/) (installation instructions available on the official PyTorch website)
- [torchvision](https://pytorch.org/vision/stable/index.html) (PyTorch's package for computer vision tasks)
- [Ultralytics](https://github.com/ultralytics/yolov5) (installation instructions available on the Ultralytics GitHub repository: https://github.com/ultralytics/ultralytics)

You can also use the following command to install all required libraries.
```bash
pip install -r requirements. txt
```


## Usage
When trying to reproduce the result of this project, it is important to set the [config.yaml](config.yaml) file first.  
After changing the root directory pointing to your data directory, you are good to go.

```bash
# Train the model
python yolo_train.py

# Validate the model
python yolo_val.py

# Run the model against real-time CS-2 game
python yolo_detect.py
```
