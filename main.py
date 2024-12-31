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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_cifar_data() # domyślne ustawienia, 50 klas
    print(f"Number of classes: {len(class_names)}")
    
    with open('class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))

    print("Initializing model...")
    # model = ResNet(
    #     version=50
    #     num_classes=50,  
    #     pretrained=True,
    #     layers_to_unfreeze=2,
    #     expand=True
    # )
    
    model = WideResNet(
        depth=28,
        widen_factor=10,
        dropout_rate=0.3,
        num_classes=50
    )

    print("Setting up trainer...")
    config = TrainerConfig(
        epochs=200,
        learning_rate=0.1, # starting lr for scheduler
        weight_decay=5e-4,
        checkpoint_dir='./checkpoints',
        scheduler='wide_resnet',
        mixup=False
    )
    
    trainer = ModelTrainer(model, device, config)
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, 'last_checkpoint.pth')
    if os.path.exists(checkpoint_path):
        print("Restoring checkpoint...")
        trainer.load_checkpoint(checkpoint_path)

    try:
        print("Starting training...")
        trainer.train(train_loader, val_loader)

        print("\nEvaluating on test set...")
        best_checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        trainer.load_checkpoint(best_checkpoint_path)
        
        test_metrics, test_accuracy = trainer.evaluate(test_loader, phase='test')
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"Test Loss: {test_metrics['test_loss']:.4f}")

    finally:
        trainer.cleanup()


if __name__ == '__main__':
    main()