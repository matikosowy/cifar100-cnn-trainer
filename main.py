from cifar100cnn.models import ResNet, WideResNet
from cifar100cnn.train import ModelTrainer, TrainerConfig
from cifar100cnn.data import get_cifar_data
import torch
import numpy as np
import random
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# SETTINGS
MODE = 'train' # train or inference
RESUME_TRAINING = False # resume training from last checkpoint
MODEL_FOR_INFERENCE = 'checkpoints/resnet18/fine-tuned/best_model.pth'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_cifar_data(num_classes=50, augment=True)
    print(f"Number of classes: {len(class_names)}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    print("Initializing model...")
    # model = ResNet(
    #     version=50,
    #     num_classes=50,  
    #     pretrained=True,
    #     layers_to_unfreeze=2,
    # )
    
    model = WideResNet(
        depth=28,
        widen_factor=10,
        dropout_rate=0.5,
        num_classes=50
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if model.name.startswith('resnet'):
        name_suffix = 'from-scratch' if not model.pretrained else 'fine-tuned'
    else:
        name_suffix = None

    print("Setting up trainer...")
    config = TrainerConfig(
        epochs=100,
        learning_rate=0.1, # starting lr for scheduler
        weight_decay=5e-4,
        checkpoint_dir=f'checkpoints/{model.name}/{name_suffix}' if name_suffix else f'checkpoints/{model.name}',
        experiment_name=f'{model.name}_{name_suffix}',
        scheduler='cos',
        mixup=False,
        label_smoothing=0.1,
    )
    
    trainer = ModelTrainer(model, device, config, class_names=class_names, train_loader=train_loader)
    
    if MODE == 'train':
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(config.checkpoint_dir, 'last_checkpoint.pth')
        
        if RESUME_TRAINING and os.path.exists(checkpoint_path):
            print("Restoring checkpoint...")
            trainer.load_checkpoint(checkpoint_path)

        try:
            # Training
            print("Starting training...")
            try:
                trainer.train(train_loader, val_loader)
            except ValueError as e:
                error_message = str(e)
                if "Tried to step" in error_message and RESUME_TRAINING:
                    print(f"Caught ValueError: {error_message}. LR reset.")
                    trainer.load_checkpoint(checkpoint_path, reset_lr=True)
                    trainer.train(train_loader, val_loader)
                else:
                    raise e
                    
            # Inference on test set after training
            print("Loading best model...")
            best_checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            trainer.load_checkpoint(best_checkpoint_path)
            
            print("\nEvaluating on test set...")
            test_metrics, test_accuracy = trainer.evaluate(test_loader, phase='test')
            print(f"Test Accuracy: {test_accuracy*100:.2f}%")
            print(f"Test Loss: {test_metrics['test_loss']:.4f}")

        finally:
            trainer.cleanup()
            
    if MODE == 'inference':
        try:
            trainer.load_checkpoint(MODEL_FOR_INFERENCE, inference=True)
            
            print("Evaluating on test set...")
            test_metrics, test_accuracy = trainer.evaluate(test_loader, phase='test', inference=True)
            print(f"Test Accuracy: {test_accuracy*100:.2f}%")
            print(f"Test Loss: {test_metrics['test_loss']:.4f}")
        
        finally:
            trainer.cleanup()


if __name__ == '__main__':
    main()