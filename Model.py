import torch
import torch.nn as nn
from torchvision import models
from config import S, B, C, IMG_size

import warnings
warnings.filterwarnings("ignore")

class YOLOv1_Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        output_dim = S * S * (B * 5 + C)

        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        self.backbone = nn.Sequential(*features)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((S, S))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * S* S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B*5 + self.C)

        return x
    
# if __name__ == "__main__":
#     model = YOLOv1_Model()
#     print(model)
#     inp = torch.randn(2, 3, IMG_size, IMG_size)
#     out = model(inp)
#     print("out shape:", out.shape)