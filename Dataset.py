import os
import torch
from torch.utils.data import Dataset
from config import Data_dir, IMG_size, S, C, B
from torchvision import transforms
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

VOC_classes = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa",
    "train","tvmonitor"
]
class_to_id = {c: i for i,c in enumerate(VOC_classes)}

transform = transforms.Compose([
    transforms.Resize((IMG_size, IMG_size)),
    transforms.ToTensor(),
])

def Prase_VOC_XML(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    Size = root.find('size')
    w = int(Size.find('width').text)
    h = int(Size.find('height').text)
    boxes = []
    labels = []
    for obj in root.findall('object'):
        name = obj.find("name").text.lower().strip()
        bbox = obj.find("bndbox")
        x_min = float(bbox.find("xmin").text)
        y_min = float(bbox.find("ymin").text)
        x_max = float(bbox.find("xmax").text)
        y_max = float(bbox.find("ymax").text)

        boxes.append([x_min,y_min,x_max,y_max])
        labels.append(name)

    return w, h, boxes, labels

def Boxes_to_YOLO_target(boxes, labels, w, h):
    target = np.zeros((S,S,5 + C))
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        x_center = ((x_max + x_min) / 2) / w
        y_center = ((y_max + y_min) / 2) / h
        box_w = (x_max - x_min) / w
        box_h = (y_max - y_min) / h

        i = int(x_center * S)
        j = int(y_center * S)

        if i>=S: i = S-1
        if j>=S: j = S-1

        x_cell = x_center * S - i
        y_cell = y_center * S - j

        if target[j, i, 0] == 1:
            continue

        target[j, i, 0] = 1
        target[j, i, 1:5] = np.array([x_cell, y_cell, box_w, box_h])
        class_id = class_to_id[label]
        target[j, i, 5 + class_id] = 1
    
    return target


class VOCdataset(Dataset):
    def __init__(self, voc_path=Data_dir, year="2007", imageset="trainval", transform=transform):
        self.voc_root = voc_path
        annotation_path = os.path.join(voc_path,"VOC{}\\Annotations".format(year)) #D:\DLCV_AI\YOLO\YOLOv1\VOCdevkit\VOC\VOC2007\Annotations
        img_path = os.path.join(voc_path,"VOC{}\\JPEGImages".format(year)) #D:\DLCV_AI\YOLO\YOLOv1\VOCdevkit\VOC\VOC2007\JPEGImages
        img_set_file = os.path.join(voc_path,"VOC{}\\ImageSets\\Main\\{}.txt".format(year, imageset))#D:\DLCV_AI\YOLO\YOLOv1\VOCdevkit\VOC\VOC2007\ImageSets\Main
        with open(img_set_file) as f:
            image_ids = [x.strip() for x in f.readlines()]
        self.items = []
        for img_id in image_ids:
            img = os.path.join(img_path,"{}.jpg".format(img_id))
            annotation = os.path.join(annotation_path,"{}.xml".format(img_id))
            if os.path.exists(img) and os.path.exists(annotation):
                self.items.append((img, annotation))
        self.transform = transform

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        img, xml = self.items[index]
        w, h, boxes, labels = Prase_VOC_XML(xml)
        target = Boxes_to_YOLO_target(boxes, labels, w, h)
        image = Image.open(img).convert("RGB")
        image_tensor = self.transform(image)

        return image_tensor, torch.from_numpy(target)
