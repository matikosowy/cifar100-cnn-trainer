import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR, ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime
import wandb
from pathlib import Path


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
    """Klasa konfiguracyjna dla trenera modelu.
    
    Args:
        epochs (int): Liczba epok.
        learning_rate (float): Współczynnik uczenia.
        weight_decay (float): Współczynnik regularyzacji L2.
        label_smoothing (float): Współczynnik wygładzania etykiet.
        checkpoint_dir (str): Ścieżka do zapisu checkpointów.
        scheduler (str): Typ schedulera (cos, wide_resnet, one_cycle, reduce).
        mixup (bool): Czy używać Mixup.
        project_name (str): Nazwa projektu w Weight & Biases.
        experiment_name (str): Nazwa eksperymentu w Weight & Biases.
        log_predictions_freq (int): Częstotliwość logowania predykcji.
        tags (list): Lista tagów dla eksperymentu.
    """
    def __init__(
        self,
        epochs = 200,
        learning_rate = 0.1,
        weight_decay = 5e-4,
        label_smoothing = 0.1,
        checkpoint_dir = './checkpoints',
        scheduler = 'cos',
        mixup=True,
        project_name='cifar100-cnn',
        experiment_name=None,
        log_predictions_freq=100,
        tags=None
    ):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir
        self.scheduler = scheduler
        self.label_smoothing = label_smoothing
        self.mixup = mixup
        self.project_name = project_name
        self.experiment_name = experiment_name or datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_predictions_freq = log_predictions_freq
        self.tags = tags or []

        os.makedirs(checkpoint_dir, exist_ok=True)


class ModelTrainer:
    """Klasa trenera modelu."""
    def __init__(self, model, device, config, train_loader=None, class_names=None):
        self.device = device
        self.model = model.to(device)
        self.config = config
        self.class_names = class_names
        
        self.wandb_step = 0
        
        self.init_wandb()
        wandb.watch(model, log='all', log_freq=config.log_predictions_freq)
        
        self.criterion_train = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.criterion_eval = nn.CrossEntropyLoss()
        
        self.train_loader = train_loader
        self.optimizer = None
        self.scheduler = None
        self._init_optimizer_and_scheduler(config, train_loader)
        
        self.scaler = GradScaler() if device.type == 'cuda' else None
        self.best_accuracy = 0
        self.current_epoch = 0
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
    def init_wandb(self):
        """Inicjalizacja Weight & Biases."""
        config_dict = {
            "learning_rate": self.config.learning_rate,
            "epochs": self.config.epochs,
            "weight_decay": self.config.weight_decay,
            "label_smoothing": self.config.label_smoothing,
            "scheduler": self.config.scheduler,
            "mixup": self.config.mixup,
            "architecture": self.model.__class__.__name__,
            "optimizer": "SGD",
            "device": self.device.type
        }
        
        wandb.init(
            project=self.config.project_name,
            name=self.config.experiment_name,
            config=config_dict,
            tags=self.config.tags,
            save_code=True
        )

    def log_batch_predictions(self, inputs, outputs, labels, prefix="train"):
        """Zapisuje predykcje modelu do W&B."""
        _, predicted = outputs.max(1)
        
        num_images = min(8, inputs.size(0))
        images = inputs[:num_images]
        preds = predicted[:num_images]
        actual = labels[:num_images]
        
        img_grid = torchvision.utils.make_grid(images, normalize=True)
        
        img_grid_np = img_grid.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
        
        if self.class_names is not None:
            captions = [
            f"Pred: {self.class_names[p.item()]}, True: {self.class_names[a.item()]}"
            for p, a in zip(preds, actual)
            ]
        else:
            print("No class names provided, using class indices")
            captions = [f"Pred: {p.item()}, True: {a.item()}" for p, a in zip(preds, actual)]
        
        wandb.log({
            f"{prefix}/predictions": wandb.Image(
                img_grid_np,
                caption=f"Batch Predictions - {prefix}\n" + "\n".join(captions)
            )
        }, step=self.wandb_step)

    def log_confusion_matrix(self, all_preds, all_targets, num_classes, prefix="train"):
        """Zapisuje confusion matrix do W&B."""
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        for pred, target in zip(all_preds, all_targets):
            confusion_matrix[target, pred] += 1
            
        wandb.log({
            f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_targets.cpu().numpy(),
                preds=all_preds.cpu().numpy(),
                class_names=self.class_names
            )
        }, step=self.wandb_step)


    def train_epoch(self, train_loader, epoch):
        """Trening modelu przez jedną epokę."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.epochs}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            if self.config.mixup:
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
                
            if self.config.scheduler == 'one_cycle':
                self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if not self.config.mixup:
                all_preds.extend(predicted.cpu())
                all_targets.extend(labels.cpu())
            
            batch_metrics = {
                "train/batch_loss": loss.item(),
                "train/batch_accuracy": 100. * correct / total,
                "train/learning_rate": self.optimizer.param_groups[0]['lr']
            }
            wandb.log(batch_metrics, step=self.wandb_step)
            
            if batch_idx % self.config.log_predictions_freq == 0 and not self.config.mixup:
                self.log_batch_predictions(inputs, outputs, labels, prefix="train")
            
            self.wandb_step += 1
            
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

    def evaluate(self, val_loader, phase='val', inference=False):
        """Walidacja modelu."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc=f'{phase.capitalize()} Evaluation')):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion_eval(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu())
                all_targets.extend(labels.cpu())
                
                if batch_idx % self.config.log_predictions_freq == 0 and not inference:
                    self.log_batch_predictions(inputs, outputs, labels, prefix=phase)
                    
                self.wandb_step += 1
        
        if not inference:  
            self.log_confusion_matrix(
                torch.tensor(all_preds),
                torch.tensor(all_targets),
                outputs.size(1),
                prefix=phase
            )

        accuracy = correct / total
        metrics = {
            f'{phase}_loss': running_loss / len(val_loader),
            f'{phase}_acc': 100. * accuracy
        }
        
        return metrics, accuracy

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint and log to W&B"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'metrics': metrics,
            'scheduler_type': self.config.scheduler
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir)
        last_checkpoint_path = checkpoint_path / 'last_checkpoint.pth'
        best_checkpoint_path = checkpoint_path / 'best_model.pth'
        
        torch.save(checkpoint, last_checkpoint_path)
        if is_best:
            torch.save(checkpoint, best_checkpoint_path)
        
        if is_best:
            best_artifact = wandb.Artifact(
                name="best-model",
                type="model",
                description=f"Best model checkpoint (epoch {epoch})"
            )
            best_artifact.add_file(str(best_checkpoint_path))
            wandb.log_artifact(best_artifact)

    def load_checkpoint(self, checkpoint_path, inference=False, reset_lr=False):
        """Wczytuje stan modelu, optymalizatora i schedulera z checkpointu."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if not any(key.startswith('model.') for key in checkpoint['model_state_dict'].keys()):
            model_state_dict = {f'model.{k}': v for k, v in checkpoint['model_state_dict'].items()}
        else:
            model_state_dict = checkpoint['model_state_dict']
    
        self.model.load_state_dict(model_state_dict)
        
        if not inference:
            saved_scheduler_type = checkpoint['scheduler_type'] if 'scheduler_type' in checkpoint else None
            current_scheduler_type = self.config.scheduler
            
            saved_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
            
            should_reset = reset_lr or (saved_scheduler_type != current_scheduler_type) or (saved_lr < 1e-6)
            
            if should_reset:
                print(
                    f"Resetting optimizer and scheduler..."
                    f"Saved scheduler type: {saved_scheduler_type}, current scheduler type: {current_scheduler_type}"
                )

                self._init_optimizer_and_scheduler(self.config, self.train_loader)
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            self.best_accuracy = checkpoint['best_accuracy']   
            return checkpoint['epoch']
        else:
            return None

    def train(self, train_loader, val_loader):
        """Trening modelu."""
        try:
            best_val_acc = 0.0
            
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Faza trenowania
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Faza walidacji
                val_metrics, accuracy = self.evaluate(val_loader)
                val_loss = val_metrics['val_loss']
                
                if self.config.scheduler != 'one_cycle':
                    if self.config.scheduler == 'reduce':
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                        
                epoch_metrics = {
                    "epoch": epoch,
                    "train/epoch_loss": train_metrics['train_loss'],
                    "train/epoch_accuracy": train_metrics['train_acc'],
                    "val/epoch_loss": val_metrics['val_loss'],
                    "val/epoch_accuracy": val_metrics['val_acc'],
                    "learning_rate": train_metrics['learning_rate']
                }
                
                wandb.log(epoch_metrics, step=self.wandb_step)
                
                print(f'\nEpoch {epoch+1}/{self.config.epochs}:')
                print(f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
                print(f'Training {"(mixup)" if self.config.mixup else ""} - loss: {train_metrics["train_loss"]:.4f}, '
                      f'acc: {train_metrics["train_acc"]:.2f}%')
                print(f'Validation - loss: {val_loss:.4f}, acc: {accuracy*100:.2f}%')

                # Zapisz najlepszy model
                is_best = accuracy > best_val_acc
                if is_best:
                    best_val_acc = accuracy
                    print(f'New best model saved!')
                
                if is_best or epoch % 10 == 0:
                    self.save_checkpoint(
                        epoch, 
                        {**train_metrics, **val_metrics},
                        is_best=is_best
                    )
                self.wandb_step += 1
                    
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
                
    def _init_optimizer_and_scheduler(self, config, train_loader=None):
        """Inicjalizacja optymalizatora i schedulera."""
        if config.scheduler not in ['cos', 'wide_resnet', 'one_cycle', 'reduce']:
            raise ValueError("Invalid scheduler type! Use 'cos', 'wide_resnet', 'one_cycle' or 'reduce'")
        
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
            
        if config.scheduler == 'one_cycle':
            if train_loader is None:
                raise ValueError("train_loader is required for OneCycleLR scheduler")
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=config.learning_rate,
                steps_per_epoch=len(train_loader),
                epochs=config.epochs,
            )
            
        if config.scheduler == 'reduce':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
            )

    # Zwalnianie zasobów
    def cleanup(self):
        print("Starting cleanup...")
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        try:
            if wandb.run is not None:
                print("Finishing wandb run...")
                wandb.finish()
        except Exception as e:
            print(f"Warning: Error during wandb cleanup: {str(e)}")
            try:
                wandb.finish(exit_code=1)
            except:
                print("Failed to force close wandb")
        
        print("Cleanup completed")


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