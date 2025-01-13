import torch 
import torch.nn as nn 

class Emotic(nn.Module):
  ''' Emotic Model'''
  def __init__(self, num_context_features, num_body_features):
    super(Emotic,self).__init__()
    self.num_context_features = num_context_features
    self.num_body_features = num_body_features
    
    # 定义特征转换层 (ResNet18的特征维度是512)
    self.resnet_full_transform = nn.Linear(512, 256)
    self.resnet_bbox_transform = nn.Linear(512, 256)
    
    # 定义融合层和分类器
    self.fusion = nn.Sequential(
      nn.Linear(512, 512),  # 两个256维特征的拼接
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(256, 26),
      nn.Sigmoid()
    )

    
  def forward(self, x_context, x_body):
    context_features = x_context.view(-1, self.num_context_features)
    body_features = x_body.view(-1, self.num_body_features)

    # 转换特征维度
    resnet_full_features = self.resnet_full_transform(context_features)
    resnet_bbox_features = self.resnet_bbox_transform(body_features)
    
    # 特征融合
    combined_features = torch.cat([resnet_full_features, resnet_bbox_features], dim=1)
    
    # 通过融合层和分类器
    output = self.fusion(combined_features)
    
    return output
