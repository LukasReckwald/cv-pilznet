"""Baut ImageNet-vortrainierte CNNs für Transfer Learning auf den Pilzdatensatz."""

import torch.nn as nn
from torchvision.models import (
    MobileNet_V2_Weights,
    ResNet50_Weights,
    mobilenet_v2,
    resnet50,
)

ARCHITECTURES = ("mobilenet_v2", "resnet50")


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet_v2":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unbekannte Architektur '{arch}'. Erlaubt: {ARCHITECTURES}")


def set_backbone_trainable(model: nn.Module, arch: str, trainable: bool) -> None:
    """Friert alle Parameter außer dem neuen Klassifikations-Head ein/entfriert sie.

    Erst wird trainable auf ALLE Parameter gesetzt, danach wird der Head
    wieder auf "trainierbar" zurückgesetzt - so bleibt er immer aktiv,
    egal ob das Backbone gerade eingefroren ist oder nicht.
    """
    head = model.classifier[-1] if arch == "mobilenet_v2" else model.fc
    for param in model.parameters():
        param.requires_grad = trainable
    for param in head.parameters():
        param.requires_grad = True
