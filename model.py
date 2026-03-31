import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.feature_channels = self.backbone.feature_info.channels()
    
    def forward(self, x):
        return self.backbone(x)


class SegFormerDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes=1, decoder_dim=256):
        super().__init__()
        self.proj = nn.ModuleList()
        for in_ch in encoder_channels:
            self.proj.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, decoder_dim, kernel_size=1),
                    nn.BatchNorm2d(decoder_dim),
                    nn.ReLU(inplace=True)
                )
            )
        self.fusion = nn.Sequential(
            nn.Conv2d(decoder_dim * len(encoder_channels), decoder_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        )
    
    def forward(self, features):
        target_size = features[0].shape[2]
        projected = []
        for i, feat in enumerate(features):
            proj = self.proj[i](feat)
            if proj.shape[2] != target_size:
                proj = F.interpolate(proj, size=(target_size, target_size), 
                                     mode='bilinear', align_corners=False)
            projected.append(proj)
        concat = torch.cat(projected, dim=1)
        out = self.fusion(concat)
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        return out


class LungUltrasoundModel(nn.Module):
    def __init__(self, num_classes=3, num_seg_classes=1):
        super().__init__()
        self.encoder = EfficientNetEncoder('efficientnet_b3', pretrained=False)
        encoder_channels = self.encoder.feature_channels
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[-1], 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.seg_decoder = SegFormerDecoder(encoder_channels, num_seg_classes, 256)
    
    def forward(self, x):
        features = self.encoder(x)
        class_out = self.classifier(features[-1])
        seg_out = self.seg_decoder(features)
        return class_out, seg_out