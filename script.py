import argparse
import torch
import numpy as np
import random
import os
from cifar100cnn.models import ResNet, WideResNet
from cifar100cnn.train import ModelTrainer, TrainerConfig
from cifar100cnn.data import get_cifar_data

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train or evaluate a CIFAR-100 CNN model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    main_group = parser.add_argument_group('Main functionality')
    # Required arguments
    main_group.add_argument(
        '--mode', 
        choices=['train', 'inference'],
        required=True,
        help='Mode: train a new model or run inference on an existing one'
    )
    
    main_group.add_argument(
        '--model',
        choices=['resnet', 'wideresnet'],
        required=True,
        help='Model architecture to use'
    )
    
    # Training configuration
    training_group = parser.add_argument_group('Training configuration')
    training_group.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    training_group.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Initial learning rate'
    )
    
    training_group.add_argument(
        '--wd',
        type=float,
        default=5e-4,
        help='Weight decay (L2 regularization)'
    )
    
    training_group.add_argument(
        '--scheduler',
        choices=['adam', 'sgd', 'cos', 'reduce', '1cycle'],
        default='cos',
        help='Learning rate scheduler type'
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model configuration')
    model_group.add_argument(
        '--classes',
        type=int,
        default=50,
        help='Number of output classes (max 100)'
    )
    
    model_group.add_argument(
        '--version',
        type=int,
        default=50,
        help='ResNet version (18 or 50)'
    )
    
    model_group.add_argument(
        '--pretrained',
        action='store_true',
        help='Use pretrained weights for ResNet',
    )
    
    model_group.add_argument(
        '--unfreeze',
        type=int,
        default=2,
        help='Number of layers to unfreeze for fine-tuning'
    )
    
    # Augmentation options
    regularization_group = parser.add_argument_group('Regularization options')
    regularization_group.add_argument(
        '--mixup',
        action='store_true',
        help='Enable mixup augmentation'
    )
    
    regularization_group.add_argument(
        '--ls',
        type=float,
        default=0.1,
        help='Label smoothing factor (0-1)'
    )
    
    regularization_group.add_argument(
        '--augment',
        action='store_true',
        help='Enable standard data augmentation (crop, flip, rotate, etc.)'
    )
    
    # Checkpoint handling
    checkpoint_group = parser.add_argument_group('Checkpoint handling')
    checkpoint_group.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    
    checkpoint_group.add_argument(
        '--path',
        type=str,
        help='Path to model checkpoint (required for inference)'
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == 'inference' and not args.path:
        parser.error("--path is required when mode is 'inference'")
    
    if args.classes > 100:
        parser.error("--classes cannot exceed 100 (CIFAR-100 limitation)")
    
    if not (0 <= args.ls <= 1):
        parser.error("--ls must be between 0 and 1")
    
    if args.model == 'resnet':
        if args.version not in [18, 50]:
            parser.error("--version must be 18 or 50")
        
        if args.pretrained and (args.unfreeze < 0 or args.unfreeze > 5):
            parser.error("--unfreeze must be between 0 and 5 when using pretrained weights")
        
        
    return args

def main():
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = get_cifar_data(num_classes=args.classes, augment=args.augment)
    print(f"Number of classes: {len(class_names)}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Validation size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    print("Initializing model...")
    if args.model == 'resnet':
        model = ResNet(
            version=args.version,
            num_classes=args.classes,
            pretrained=args.pretrained,
            layers_to_unfreeze=args.unfreeze,
        )
    else:
        model = WideResNet(
            depth=28,
            widen_factor=10,
            dropout_rate=0.5,
            num_classes=args.classes
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    name_suffix = 'fine-tuned' if args.model == 'resnet' and model.pretrained else 'from-scratch'
    checkpoint_dir = f'checkpoints/{model.name}/{name_suffix}'

    print("Setting up trainer...")
    config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        checkpoint_dir=checkpoint_dir,
        experiment_name=f'{model.name}_{name_suffix}',
        scheduler=args.scheduler,
        mixup=args.mixup,
        label_smoothing=args.ls,
    )

    trainer = ModelTrainer(model, device, config, class_names=class_names, train_loader=train_loader)

    if args.mode == 'train':
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(config.checkpoint_dir, 'last_checkpoint.pth')

        if args.resume and os.path.exists(checkpoint_path):
            print("Restoring checkpoint...")
            trainer.load_checkpoint(checkpoint_path)

        try:
            print("Starting training...")
            try:
                trainer.train(train_loader, val_loader)
            except ValueError as e:
                if "Tried to step" in str(e) and args.resume:
                    print(f"Caught ValueError: {str(e)}. Resetting LR and resuming training.")
                    trainer.load_checkpoint(checkpoint_path, reset_lr=True)
                    trainer.train(train_loader, val_loader)
                else:
                    raise e

            print("Loading best model...")
            best_checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
            trainer.load_checkpoint(best_checkpoint_path)

            print("\nEvaluating on test set...")
            test_metrics, test_accuracy = trainer.evaluate(test_loader, phase='test')
            print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
            print(f"Test Loss: {test_metrics['test_loss']:.4f}")
        finally:
            trainer.cleanup()
    
    if args.mode == 'inference':
        try:
            trainer.load_checkpoint(args.path, inference=True)
            print("Evaluating on test set...")
            test_metrics, test_accuracy = trainer.evaluate(test_loader, phase='test', inference=True)
            print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
            print(f"Test Loss: {test_metrics['test_loss']:.4f}")
        finally:
            trainer.cleanup()

if __name__ == '__main__':
    main()
