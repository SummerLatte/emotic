import torch 
import torch.nn as nn 
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.query = nn.Linear(dim1, 256)
        self.key = nn.Linear(dim2, 256)
        self.value = nn.Linear(dim2, 256)
        self.scale = 256 ** -0.5

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        out = torch.matmul(attention, v)
        return out

class Emotic(nn.Module):
    def __init__(self, num_context_features, num_body_features, model_size='large'):
        super(Emotic,self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        
        # 初始化BLIP模型
        if model_size == 'large':
            self.model_name = "Salesforce/blip-image-captioning-large"
            self.blip_hidden_size = 1024
        else:
            self.model_name = "Salesforce/blip-image-captioning-base"
            self.blip_hidden_size = 768
        
        # 直接初始化BLIP模型和处理器
        self.blip = None
        self.processor = None
        self._initialize_blip()
        
        # 初始化ResNet模型用于处理context特征
        self.resnet_context = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_context = nn.Sequential(*(list(self.resnet_context.children())[:-1]))
        
        # 初始化ResNet模型用于处理body特征
        self.resnet_body = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_body = nn.Sequential(*(list(self.resnet_body.children())[:-1]))
        
        # 初始化ResNet模型用于处理face特征
        self.resnet_face = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_face = nn.Sequential(*(list(self.resnet_face.children())[:-1]))
        
        # 定义特征转换层
        self.blip_transform = nn.Linear(self.blip_hidden_size, 256)
        self.resnet_context_transform = nn.Linear(512, 256)
        self.resnet_body_transform = nn.Linear(512, 256)
        self.resnet_face_transform = nn.Linear(512, 256)
        
        # 添加自注意力模块
        self.self_attention_blip = SelfAttention(256)
        self.self_attention_context = SelfAttention(256)
        self.self_attention_body = SelfAttention(256)
        self.self_attention_face = SelfAttention(256)
        
        # 添加交叉注意力模块
        self.cross_attention_context_body = CrossAttention(256, 256)
        self.cross_attention_context_face = CrossAttention(256, 256)
        self.cross_attention_blip_context = CrossAttention(256, 256)
        
        # 修改融合层
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 26),
            nn.Sigmoid()
        )

    def _initialize_blip(self):
        """初始化BLIP模型，确保在主进程中只初始化一次"""
        if self.blip is None:
            self.blip = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            # 冻结BLIP的参数
            for param in self.blip.parameters():
                param.requires_grad = False

    def __getstate__(self):
        """自定义序列化方法"""
        state = self.__dict__.copy()
        # 移除无法序列化的BLIP组件
        state['blip'] = None
        state['processor'] = None
        return state

    def __setstate__(self, state):
        """自定义反序列化方法"""
        self.__dict__.update(state)
        # 重新初始化BLIP组件
        self._initialize_blip()

    def forward(self, x_context, x_body, x_face, has_face):
        # 确保BLIP模型在正确的设备上
        device = x_context.device
        if self.blip is None:
            self._initialize_blip()
        if next(self.blip.parameters()).device != device:
            self.blip = self.blip.to(device)
        
        # 处理context图像的BLIP特征
        with torch.no_grad():
            batch_size = x_context.size(0)
            pil_images = []
            for i in range(batch_size):
                img = x_context[i].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                pil_images.append(pil_img)
            
            inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
            pixel_values = inputs.pixel_values.to(device)
            
            vision_outputs = self.blip.vision_model(pixel_values)
            blip_features = vision_outputs.pooler_output

        # 使用ResNet处理context特征
        context_features = self.resnet_context(x_context)
        context_features = context_features.view(context_features.size(0), -1)
        
        # 使用ResNet处理body特征
        body_features = self.resnet_body(x_body)
        body_features = body_features.view(body_features.size(0), -1)
        
        # 使用ResNet处理face特征
        face_features = self.resnet_face(x_face)
        face_features = face_features.view(face_features.size(0), -1)
        
        # 根据has_face调整face_features的权重
        has_face = has_face.view(-1, 1)
        face_features = face_features * has_face
        
        # 转换特征维度
        blip_features = self.blip_transform(blip_features)
        context_features = self.resnet_context_transform(context_features)
        body_features = self.resnet_body_transform(body_features)
        face_features = self.resnet_face_transform(face_features)
        
        # 应用自注意力
        blip_features = blip_features.unsqueeze(1)  # [B, 1, 256]
        context_features = context_features.unsqueeze(1)
        body_features = body_features.unsqueeze(1)
        face_features = face_features.unsqueeze(1)
        
        blip_att = self.self_attention_blip(blip_features)
        context_att = self.self_attention_context(context_features)
        body_att = self.self_attention_body(body_features)
        face_att = self.self_attention_face(face_features)
        
        # 应用交叉注意力
        context_body_att = self.cross_attention_context_body(context_att, body_att)
        context_face_att = self.cross_attention_context_face(context_att, face_att)
        blip_context_att = self.cross_attention_blip_context(blip_att, context_att)
        
        # 特征融合
        context_fused = (context_body_att + context_face_att) / 2
        final_features = torch.cat([
            blip_context_att.squeeze(1),
            context_fused.squeeze(1),
            body_att.squeeze(1),
            face_att.squeeze(1)
        ], dim=1)
        
        # 通过融合层
        output = self.fusion(final_features)
        
        return output
