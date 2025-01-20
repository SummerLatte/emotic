import torch 
import torch.nn as nn 
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim2
        self.head_dim = dim2 // num_heads
        assert self.head_dim * num_heads == dim2, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim1, dim2)
        self.k_proj = nn.Linear(dim2, dim2)
        self.v_proj = nn.Linear(dim2, dim2)
        self.proj = nn.Linear(dim2, dim2)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim2)
        self.norm2 = nn.LayerNorm(dim2)
        self.mlp = nn.Sequential(
            nn.Linear(dim2, dim2 * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim2 * 2, dim2)
        )

    def forward(self, x1, x2):
        B, N1, C1 = x1.shape
        B, N2, C2 = x2.shape
        shortcut = self.q_proj(x1)  # 将x1投影到与x2相同的维度
        
        # Layer Norm
        x1 = self.norm1(shortcut)
        x2 = self.norm1(x2)
        
        # Multi-head Cross Attention
        q = self.q_proj(x1).reshape(B, N1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x2).reshape(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x2).reshape(B, N2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N1, self.dim)
        x = self.proj(x)
        x = self.dropout(x)
        
        # Residual connection
        x = x + shortcut
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        return x

class FeatureAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, x):
        # x: [B, N, C]
        attn_weights = self.attention(x)  # [B, N, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_feature = (x * attn_weights).sum(dim=1)  # [B, C]
        return weighted_feature

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=2, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True),
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim),
                'mlp': nn.Sequential(
                    nn.Linear(dim, dim * mlp_ratio),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * mlp_ratio, dim)
                )
            }) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.layers:
            # Self-attention
            attn_out, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # MLP
            mlp_out = layer['mlp'](x)
            x = layer['norm2'](x + mlp_out)
        return x

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
        
        # 初始化单个共享的ResNet模型
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))  # 移除最后的全连接层
        # 冻结ResNet参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # 定义特征转换层
        self.blip_transform = nn.Linear(self.blip_hidden_size, 128)
        self.resnet_context_transform = nn.Linear(512, 128)  # ResNet特征维度是512
        self.resnet_body_transform = nn.Linear(512, 128)
        self.resnet_face_transform = nn.Linear(512, 128)
        
        # 添加交叉注意力模块
        self.cross_attention_blip_context = CrossAttention(128, 128)
        
        # 添加特征聚合模块
        self.blip_context_aggregation = FeatureAggregation(128)
        
        # 修改融合层为Transformer结构
        self.feature_dim = 128
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        
        # Transformer编码器
        self.transformer = TransformerEncoder(
            dim=self.feature_dim,
            num_heads=8,
            num_layers=2,
            dropout=0.1
        )
        
        # 最终分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 26),
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
            blip_features = vision_outputs.last_hidden_state  # 使用序列特征 [B, L, C]
            
        # 直接使用ResNet提取特征
        with torch.no_grad():
            context_features = self.resnet(x_context).squeeze(-1).squeeze(-1)  # [B, 512]
            body_features = self.resnet(x_body).squeeze(-1).squeeze(-1)
            face_features = self.resnet(x_face).squeeze(-1).squeeze(-1)
        
        # 转换特征维度
        blip_features = self.blip_transform(blip_features)  # [B, L, 128]
        context_features = self.resnet_context_transform(context_features).unsqueeze(1)  # [B, 1, 128]
        body_features = self.resnet_body_transform(body_features).unsqueeze(1)  # [B, 1, 128]
        face_features = self.resnet_face_transform(face_features).unsqueeze(1)  # [B, 1, 128]
        
        # 根据has_face调整face_features的权重
        has_face = has_face.view(-1, 1, 1)
        face_features = face_features * has_face
        
        # 应用交叉注意力 - 只保留BLIP和context之间的交叉注意力
        blip_context_att = self.cross_attention_blip_context(blip_features, context_features)
        
        # 特征聚合
        blip_context_att = self.blip_context_aggregation(blip_context_att)  # [B, 128]
        context_features = context_features.squeeze(1)  # [B, 128]
        body_features = body_features.squeeze(1)  # [B, 128]
        face_features = face_features.squeeze(1)  # [B, 128]
        
        # 特征融合 - 改为序列形式
        batch_size = blip_context_att.size(0)
        
        # 扩展维度使所有特征形状一致 [B, 1, C]
        blip_context_att = blip_context_att.unsqueeze(1)
        context_features = context_features.unsqueeze(1)
        body_features = body_features.unsqueeze(1)
        face_features = face_features.unsqueeze(1)
        
        # 扩展cls_token到批次大小
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # 拼接所有特征 [B, 5, C]
        sequence = torch.cat([
            cls_tokens,
            blip_context_att,
            context_features,
            body_features,
            face_features
        ], dim=1)
        
        # 通过Transformer编码器
        sequence = self.transformer(sequence)
        
        # 使用[CLS] token进行分类
        cls_feature = sequence[:, 0]
        output = self.classifier(cls_feature)
        
        return output
