# YOLOv1 From Scratch in PyTorch

This project is a simple and clean implementation of the YOLOv1 object detection model built from scratch using PyTorch.  
It includes the model architecture, YOLOv1 loss function, data preprocessing, and the training code in `train.py`.  
The goal of this project is to understand how YOLOv1 works internally and how object detection models are built step by step.

---

## What I Learned

### 1. YOLO Grid System
YOLOv1 divides the image into an S × S grid.  
Each grid cell predicts:
- One object (this is a major limitation)
- B bounding boxes
- Class probabilities

Important point:  
A grid cell can detect only one object.  
If two objects fall in the same cell, YOLOv1 will miss one of them.

### 2. Bounding Box Prediction
Each grid cell predicts B bounding boxes, and each box has:
- x and y are the center position relative to the grid cell  
- w and h are the width and height relative to the whole image  
- confidence tells how well the box matches the ground truth (IoU)

### 3. YOLOv1 Loss Function
The loss function contains:
- Localization loss for x, y, w, h  
- Confidence loss for object  
- Confidence loss for no-object  
- Classification loss  

The bounding box with the highest IoU is chosen as the "responsible" box for predicting that object.

### 4. Training Pipeline (train.py)
The training script includes:
- Loading the dataset
- Making batches
- Forward pass through the model
- Calculating YOLO loss
- Backpropagation and optimizer step
- Printing training progress

This helped me understand how to build a full training loop in PyTorch and how YOLO-style targets are prepared.

---

## Limitations of YOLOv1

1. Only one object per grid cell.  
   Objects close to each other are hard to detect.

2. Weak for small objects.  
   Small objects may not fall clearly inside one grid cell.

3. Hard to detect overlapping objects.  
   Because of the one-object-per-cell rule.

4. No anchor boxes.  
   YOLOv1 predicts raw box values, which makes training less stable than later YOLO versions.

---

## Current Project Status

Implemented so far:
- YOLOv1 model architecture  
- YOLOv1 loss function  
- Dataset preprocessing  
- Training script (`train.py`)

More features like evaluation, inference, model saving, and visualization can be added later.

---