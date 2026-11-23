"""Model zoo: U-Net (ResNet34/50) via segmentation_models_pytorch, SegFormer B0/B2/B4 via transformers.

Every wrapper returns logits at the input resolution: (B, num_classes, H, W).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAMES = ["unet_resnet34", "unet_resnet50", "segformer_b0", "segformer_b2", "segformer_b4"]


class UNetWrapper(nn.Module):
    def __init__(self, encoder: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            classes=num_classes,
        )

    def forward(self, x):
        return self.net(x)

    def encoder_params(self):
        return list(self.net.encoder.parameters())

    def head_params(self):
        enc = {id(p) for p in self.net.encoder.parameters()}
        return [p for p in self.net.parameters() if id(p) not in enc]


class SegFormerWrapper(nn.Module):
    def __init__(self, variant: str, num_classes: int, pretrained: bool = True, id2label=None):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        ckpt = f"nvidia/mit-{variant}"
        kw = {"num_labels": num_classes}
        if id2label:
            kw["id2label"] = {int(k): v for k, v in id2label.items()}
            kw["label2id"] = {v: int(k) for k, v in id2label.items()}
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                ckpt, ignore_mismatched_sizes=True, **kw
            )
        else:
            self.net = SegformerForSemanticSegmentation(SegformerConfig.from_pretrained(ckpt, **kw))

    def forward(self, x):
        logits = self.net(pixel_values=x).logits  # (B, C, H/4, W/4)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def encoder_params(self):
        return list(self.net.segformer.parameters())

    def head_params(self):
        return list(self.net.decode_head.parameters())


def build_model(name: str, num_classes: int, pretrained: bool = True, id2label=None) -> nn.Module:
    if name.startswith("unet_"):
        return UNetWrapper(name.split("_", 1)[1], num_classes, pretrained)
    if name.startswith("segformer_"):
        return SegFormerWrapper(name.split("_", 1)[1], num_classes, pretrained, id2label)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_pretrained(model: nn.Module, out_dir: str):
    """Save in the library-native format (loadable with from_pretrained)."""
    model.net.save_pretrained(out_dir)
