import os

from cifar100cnn.models import *

# paths to the best trained models
MODEL_PATHS = {
    "resnet18_scratch": "checkpoints/resnet18/from-scratch/best_model.pth",
    "resnet50_scratch": "checkpoints/resnet50/from-scratch/best_model.pth",
    "resnet18_fine_tuned": "checkpoints/resnet18/fine-tuned/best_model.pth",
    "resnet50_fine_tuned": "checkpoints/resnet50/fine-tuned/best_model.pth",
    "wide_resnet": "checkpoints/wide_resnet28_10/best_model.pth"
}

def load_model(model_name, path, device):
    """
        Loads a model from the given path.
        Returns loaded model in evaluation mode.
    
        Args:
            model_name: name of the loaded model
            path: path to the saved model.
            device: device on which the model should be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plik {path} nie istnieje.")

    if "resnet18" in model_name:
        model = ResNet(version=18, num_classes=50, pretrained=False)
    elif "resnet50" in model_name:
        model = ResNet(version=50, num_classes=50, pretrained=False)
    elif "wide_resnet" in model_name:
        model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.5, num_classes=50)
    else:
        raise ValueError(f"Nieznany model: {model_name}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.to(device)
    model.eval()

    return model