from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch.nn as nn


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet = models.resnet18(weights=None)

    def forward(self, x):
        return self.resnet(x)

def prep_models():
  # 使用两个非预训练的ResNet18，分别处理完整图像和bbox图像
  resnet_full = Resnet18()
  resnet_bbox = Resnet18()

  return resnet_full, resnet_bbox


if __name__ == '__main__':
  prep_models()


