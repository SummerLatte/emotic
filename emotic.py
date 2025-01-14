import torch 
import torch.nn as nn 
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from PIL import Image
import torchvision.models as models

class Emotic(nn.Module):
  ''' Emotic Model with BLIP'''
  def __init__(self, num_context_features, num_body_features, model_size='small'):
    super(Emotic,self).__init__()
    self.num_context_features = num_context_features
    self.num_body_features = num_body_features
    
    # 初始化BLIP模型
    if model_size == 'large':
        model_name = "Salesforce/blip-image-captioning-large"
        self.blip_hidden_size = 1024
    else:
        model_name = "Salesforce/blip-image-captioning-base"
        self.blip_hidden_size = 768
        
    # 加载BLIP模型和处理器
    self.blip = BlipForConditionalGeneration.from_pretrained(model_name)
    self.processor = BlipProcessor.from_pretrained(model_name)
    
    # 冻结BLIP的参数
    for param in self.blip.parameters():
        param.requires_grad = False
        
    # 初始化ResNet模型用于处理body特征
    self.resnet_body = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    self.resnet_body = nn.Sequential(*(list(self.resnet_body.children())[:-1]))  # 移除最后的全连接层
    
    # 定义特征转换层
    self.blip_transform = nn.Linear(self.blip_hidden_size, 256)
    self.resnet_bbox_transform = nn.Linear(512, 256)
    
    # 定义融合层和分类器 (BLIP特征 + body特征)
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
    # 处理context图像的BLIP特征
    with torch.no_grad():
        # 转换为PIL图像列表
        batch_size = x_context.size(0)
        pil_images = []
        for i in range(batch_size):
            img = x_context[i].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            pil_images.append(pil_img)
        
        # 使用BLIP处理器处理图像
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        pixel_values = inputs.pixel_values.to(x_context.device)
        
        # 获取BLIP特征
        vision_outputs = self.blip.vision_model(pixel_values)
        blip_features = vision_outputs.pooler_output

    # 使用ResNet处理body特征
    body_features = self.resnet_body(x_body)
    body_features = body_features.view(body_features.size(0), -1)

    # 转换特征维度
    blip_features = self.blip_transform(blip_features)
    resnet_bbox_features = self.resnet_bbox_transform(body_features)
    
    # 特征融合
    combined_features = torch.cat([blip_features, resnet_bbox_features], dim=1)
    
    # 通过融合层和分类器
    output = self.fusion(combined_features)
    
    return output
