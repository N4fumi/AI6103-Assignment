"""The assignment architecture, always randomly initialized."""
from torch import nn
from torchvision.models import efficientnet_v2_s


def build_model(num_classes):
    model = efficientnet_v2_s(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
