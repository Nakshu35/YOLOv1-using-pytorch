S = 7 # Grid Size (7x7)
B = 2 # number of bounding box per grid
C = 20 # number of classes 
IMG_size = 448 # image size (YOLOv1 used 448x448)

#Loss hyperparameters
L_COORD = 5.0 # weight for bounding box coordinate loss
L_NOOBJ = 0.5 # weight for confidence loss when no object

#training hyperparameters
Batch_size = 8
Learning_rate = 1e-4
epochs = 10
Weight_decay = 5e-4
Device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Paths
Data_dir = "D:\\DLCV_AI\\YOLO\\YOLOv1\\VOCdevkit\\VOC"
Save_dir = "./Checkpoints"