import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes):
    """
    Returns the specified model architecture.
    Args:
        model_name (str): Name of the model architecture ('resnet', 'vgg', 'custom').
        num_classes (int): Number of output classes.
    """
    if model_name == "resnet":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    if model_name == "custom":
        model = CustomCNN(num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

class CustomCNN(nn.Module):
    """
    A simple custom CNN for terrain classification.
    """
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
