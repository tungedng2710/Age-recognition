import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones.irse import IR_50, IR_SE_50, IR_101, IR_SE_101
from .backbones.mobilenet import MobileFaceNet
from .backbones.vit import VisionTransformer
from .backbones.convnext import convnext_tiny

class NormalizedLinear(nn.Module):
    """
    Linear layer with normalization
    """
    def __init__(self, in_features, out_features):
        super(NormalizedLinear, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        x = F.normalize(input)
        W = F.normalize(self.W)
        return F.linear(x, W)

class TonNet(nn.Module):
    def __init__(self, 
                 model_name: str = "irse", 
                 num_classes: int = 6, 
                 input_size: int = 112):
        """
        model_name (str): model name
            list of model: irse50, irse101, rexnet, mobilenet, convnext, vit
        num_classes (int): number of classes
        input_size (int): size of input sample
        """
        super().__init__()
        if "irse" in model_name:
            input_size = [input_size for i in range(2)]
            if "50" in model_name:
                self.backbone = IR_SE_50(input_size=input_size)
            else:
                self.backbone = IR_SE_101(input_size=input_size)
        elif "rexnet" in model_name:
            self.backbone = timm.create_model('rexnet_150', num_classes=num_classes)
        elif "mobilenet" in model_name:
            self.backbone = MobileFaceNet(embedding_size=512,
                                          out_h=7,
                                          out_w=7)
        elif "vit" in model_name:
            self.backbone = VisionTransformer(img_size=input_size,
                                              patch_size=8,
                                              embed_dim=512,
                                              num_heads=12,
                                              mlp_ratio=4,
                                              drop_rate=0.1)
        elif "convnext" in model_name:
            self.backbone = convnext_tiny(num_classes=512)
        else:
            print("{model_name} is not supported".format(model_name=model_name))
        # self.head = NormalizedLinear(in_features=512, out_features=num_classes)
        self.head = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, input):
        output = self.backbone(input)
        # output = self.head(output)
        return output