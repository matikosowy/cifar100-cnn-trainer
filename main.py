import os
import torch
import numpy as np
import random
from cifar100cnn.load_models import MODEL_PATHS, load_model
from cifar100cnn.models import ResNet, WideResNet
from cifar100cnn.train import ModelTrainer, TrainerConfig
from cifar100cnn.data import get_cifar_data
from cifar100cnn.extractor import *

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
        wandb.init(
            project="cifar100-knn",
            entity="matikosowy-none",
            name="knn_evaluation",
            config={
                "stage2_classes": 50,
                "stage3_classes": 50,
                "samples_per_class": [1, 5, 10],
                "n_neighbors_list": [1, 2, 5, 7, 10]
            }
        )

        train_loader, val_loader, test_loader, class_names = get_cifar_data(num_classes=100, augment=False)

        classes_A = np.arange(50)
        classes_B = np.arange(50, 100)

        models = {name: load_model(name, path, device) for name, path in MODEL_PATHS.items()}
        feature_extractor = FeatureExtractor(device)

        # stage 2:
        run_knn(
            models=models,
            feature_extractor=feature_extractor,
            train_loader=train_loader,
            test_loader=test_loader,
            classes=classes_A,
            device=device,
            stage_number=2,
            stage_name="czesc A (50 klas treningowych)",
            results_dir="cache/etap2",
            samples_per_class=[1, 5, 10],
            n_neighbors_list=[1, 2, 5, 7, 10]
        )

        # stage 3:
        run_knn(
            models=models,
            feature_extractor=feature_extractor,
            train_loader=train_loader,
            test_loader=test_loader,
            classes=classes_B,
            device=device,
            stage_number=3,
            stage_name="czesc B (50 klas nietreningowych)",
            results_dir="cache/etap3",
            samples_per_class=[1, 5, 10],
            n_neighbors_list=[1, 2, 5, 7, 10]
        )

if __name__ == '__main__':
    main()