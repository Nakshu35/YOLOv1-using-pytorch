import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Batch_size, Save_dir, Learning_rate, epochs, Weight_decay, Device
from Dataset import VOCdataset
from Model import YOLOv1_Model
from Loss import YOLOv1_Loss


def train():
    device = torch.device(Device)

    # dataset used VOC2007 trainval
    dataset = VOCdataset(imageset="trainval")
    dataloader = DataLoader(dataset, batch_size=Batch_size, shuffle=True, num_workers=2)

    model = YOLOv1_Model().to(device)
    criterion = YOLOv1_Loss()
    optimizer = optim.Adam(model.parameters(), lr=Learning_rate, weight_decay=Weight_decay)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for imgs, targets in loop:
            imgs = imgs.to(device)
            targets = targets.float().to(device)

            preds = model(imgs)

            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / (loop.n + 1))

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(Save_dir, f"yolov1_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

if __name__ == "__main__" :
    train()
