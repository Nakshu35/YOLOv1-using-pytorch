import torch
import torch.nn as nn
from config import S, B, C, L_COORD, L_NOOBJ


class YOLOv1_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = L_COORD
        self.lambda_noobj = L_NOOBJ
        self.mse = nn.MSELoss(reduction="sum")

    def compute_iou(self, box1, box2):
        # Compute IoU between box1 and box2.

        # Convert xywh -> x1, y1, x2, y2
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2

        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = area1 + area2 - inter_area + 1e-6

        return inter_area / union

    def forward(self, predictions, target):
        """
        predictions: (N, S, S, B*5 + C)
        target:      (N, S, S, 5 + C)
        """
        N = predictions.size(0)

        pred_boxes = predictions[..., : self.B * 5].view(N, self.S, self.S, self.B, 5) # (N,S,S,B,5)
        pred_cls = predictions[..., self.B * 5 :] # (N,S,S,C)

        # target components
        obj_mask = target[..., 0].unsqueeze(-1)  # (N,S,S,1)
        tgt_box = target[..., 1:5]                # (N,S,S,4)
        tgt_cls = target[..., 5:]                 # (N,S,S,C)

        # compute IoU between each predicted box and the single GT box
        tgt_box_exp = tgt_box.unsqueeze(3).expand_as(pred_boxes[..., :4])  # (N,S,S,B,4)
        ious = self.compute_iou(pred_boxes[..., :4], tgt_box_exp)  # (N,S,S,B)

        best_iou, best_idx = ious.max(dim=3, keepdim=True)  
        best_idx_expand = best_idx.unsqueeze(-1).repeat(1, 1, 1, 1, 5)  # (N,S,S,1,5)
        resp_pred_box = pred_boxes.gather(3, best_idx_expand).squeeze(3)  # (N,S,S,5)

        # COORDINATE LOSS (x,y)
        coord_loss_xy = self.mse(
            obj_mask * resp_pred_box[..., 1:3],
            obj_mask * tgt_box[..., 0:2],
        )

        # COORDINATE LOSS (w,h) with sqrt
        coord_loss_wh = self.mse(
            obj_mask * torch.sqrt(resp_pred_box[..., 3:5].clamp(min=1e-6)),
            obj_mask * torch.sqrt(tgt_box[..., 2:4].clamp(min=1e-6)),
        )

        coord_loss = self.lambda_coord * (coord_loss_xy + coord_loss_wh)

        # OBJECTNESS LOSS (for responsible boxes) - best_iou is already shape (N,S,S,1)
        obj_loss = self.mse(
            obj_mask * resp_pred_box[..., 0:1],
            obj_mask * best_iou,
        )

        # NO-OBJECT LOSS (for all boxes where there is no object in that cell)
        noobj_mask = 1 - obj_mask  # (N,S,S,1)
        pred_conf = pred_boxes[..., 0]  # (N,S,S,B)
        noobj_loss = self.lambda_noobj * self.mse(
            noobj_mask.expand_as(pred_conf) * pred_conf,
            torch.zeros_like(pred_conf),
        )

        # CLASSIFICATION LOSS (only for cells with object)
        class_loss = self.mse(
            obj_mask * pred_cls,
            obj_mask * tgt_cls,
        )

        total_loss = (coord_loss + obj_loss + noobj_loss + class_loss) / N
        return total_loss