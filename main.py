import os
import torch
import numpy as np
import random
from cifar100cnn.models import ResNet, WideResNet
from cifar100cnn.train import ModelTrainer, TrainerConfig
from cifar100cnn.data import get_cifar_data
from cifar100cnn.extractor import *

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

MODEL_PATHS = {
    "resnet18_scratch": "checkpoints/resnet18/from-scratch/best_model.pth",
    "resnet50_scratch": "checkpoints/resnet50/from-scratch/best_model.pth",
    "resnet18_fine_tuned": "checkpoints/resnet18/fine-tuned/best_model.pth",
    "resnet50_fine_tuned": "checkpoints/resnet50/fine-tuned/best_model.pth",
    "wide_resnet": "checkpoints/wide_resnet28_10/best_model.pth"
}

def load_model(model_name, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Plik {path} nie istnieje")

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

def main():
    MODE = 'knn_classification'  # 'train', 'inference', 'knn_classification'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if MODE in ['train', 'inference']:
        train_loader, val_loader, test_loader, class_names = get_cifar_data(num_classes=50, augment=True)

        print("Initializing model...")
        model = WideResNet(depth=28, widen_factor=10, dropout_rate=0.5, num_classes=50)
        
        config = TrainerConfig(
            epochs=100,
            learning_rate=0.1,
            weight_decay=5e-4,
            checkpoint_dir=f'checkpoints/{model.name}/from-scratch',
            experiment_name=f'{model.name}_from-scratch',
            scheduler='cos',
            mixup=False,
            label_smoothing=0.1,
        )
        
        trainer = ModelTrainer(model, device, config, class_names=class_names, train_loader=train_loader)
        
        if MODE == 'train':
            trainer.train(train_loader, val_loader)
        elif MODE == 'inference':
            trainer.load_checkpoint('checkpoints/wide_resnet28_10/best_model.pth', inference=True)
            test_metrics, test_accuracy = trainer.evaluate(test_loader, phase='test', inference=True)
            print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    elif MODE == 'knn_classification':
        train_loader, val_loader, test_loader, class_names = get_cifar_data(num_classes=100, augment=False)
        classes_A = np.arange(50)
        classes_B = np.arange(50, 100)

        models = {name: load_model(name, path, device) for name, path in MODEL_PATHS.items()}
        feature_extractor = FeatureExtractor(device)

        run_stage_2(models, feature_extractor, train_loader, test_loader, classes_A, device)
        run_stage_3(models, feature_extractor, train_loader, test_loader, classes_B, device)

if __name__ == '__main__':
    main()
