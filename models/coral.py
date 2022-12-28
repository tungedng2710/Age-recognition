import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__() 
        self.backbone = timm.create_model('rexnet_150', pretrained=False)
        # self.conv1 = nn.Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, input):
        return self.backbone(input)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network()
    model.eval()
    model.to(device)
    # print(model)
    dummy_input = torch.rand(4, 3, 200, 200).to(device)
    output = model(dummy_input)