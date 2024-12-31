import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR
from tqdm import tqdm
from torch.utils.data import DataLoader


def mixup_data(x, y, alpha=0.2):
    """Zwraca zmieszane dane i etykiety oraz współczynnik mieszania Mixup."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Funkcja straty Mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class TrainerConfig:
    """Klasa konfiguracyjna dla trenera modelu."""
    def __init__(
        self,
        epochs = 200,
        learning_rate = 0.1,
        weight_decay = 5e-4,
        label_smoothing = 0.1,
        checkpoint_dir = './checkpoints',
        scheduler = 'cos',
        mixup=True
    ):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.label_smoothing = label_smoothing
        self.mixup = mixup

        os.makedirs(checkpoint_dir, exist_ok=True)


class ModelTrainer:
    """Klasa trenera modelu."""
    def __init__(self, model, device, config):
        self.device = device
        self.model = model.to(device)
        self.config = config
        
        self.criterion_train = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.criterion_eval = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True
        )
        if config.scheduler == 'cos':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=config.learning_rate / 1000
            )

        if config.scheduler == 'wide_resnet':
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=[60, 120, 160],
                gamma=0.2
            )
        
        self.scaler = GradScaler() if device.type == 'cuda' else None
        self.best_accuracy = 0
        self.current_epoch = 0
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader, epoch):
        """Trening modelu przez jedną epokę."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.epochs}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            inputs, labels_a, labels_b, lam = mixup_data(
                inputs, labels
            )

            self.optimizer.zero_grad()
            
            with autocast("cuda"):
                outputs = self.model(inputs)
                
                if self.config.mixup:
                    loss = mixup_criterion(self.criterion_eval, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion_train(outputs, labels)
                
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        metrics = {
            'train_loss': running_loss / len(train_loader),
            'train_acc': 100. * correct / total,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return metrics

    def evaluate(self, val_loader, phase='val'):
        """Walidacja modelu."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'{phase.capitalize()} Evaluation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion_eval(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = correct / total
        metrics = {
            f'{phase}_loss': running_loss / len(val_loader),
            f'{phase}_acc': 100. * accuracy
        }
        
        return metrics, accuracy

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Zapisuje stan modelu, optymalizatora, schedulera i metryki."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'metrics': metrics
        }
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'best_model.pth'))
        
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, 'last_checkpoint.pth'))

    def load_checkpoint(self, checkpoint_path):
        """Wczytuje stan modelu, optymalizatora i schedulera z checkpointu."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        return checkpoint['epoch']

    def train(self, train_loader, val_loader):
        """Trening modelu."""
        try:
            best_val_loss = float('inf')
            
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Faza trenowania
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Faza walidacji
                val_metrics, accuracy = self.evaluate(val_loader)
                val_loss = val_metrics['val_loss']
                
                self.scheduler.step()
                
                print(f'\nEpoch {epoch+1}/{self.config.epochs}:')
                print(f'Training Loss: {train_metrics["train_loss"]:.4f}')
                print(f'Training Accuracy {"(mixup)" if self.config.mixup else ""}: {train_metrics["train_acc"]:.2f}%')
                print(f'Validation Loss: {val_loss:.4f}')
                print(f'Validation Accuracy: {accuracy*100:.2f}%')
                print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
                
                # Zapisz najlepszy model
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    print(f'New best validation: loss: {val_loss:.4f}, acc: {accuracy*100:.2f}%')
                
                if is_best or epoch % 10 == 0:
                    self.save_checkpoint(
                        epoch, 
                        {**train_metrics, **val_metrics},
                        is_best=is_best
                    )             
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise
        finally:
            # Zapisz ostatni checkpoint
            if self.current_epoch > 0:
                self.save_checkpoint(
                    self.current_epoch,
                    {**train_metrics, **val_metrics},
                    is_best=False
                )

    # Zwalnianie zasobów
    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_cifar_model(
    model,
    train_loader,
    val_loader,
    device = None,
    config= None
):
    """Domyślna funkcja treningu modelu CIFAR-50."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config is None:
        config = TrainerConfig()
    
    trainer = ModelTrainer(model, device, config)
    
    try:
        trainer.train(train_loader, val_loader)
    finally:
        trainer.cleanup()