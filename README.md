# CIFAR-100 CNN Training and Evaluation

Recruitment project for WUT KNSI GOLEM science club.

This repository contains code for training and evaluating Convolutional Neural Networks (CNNs) on the CIFAR-100 dataset. The project includes implementations of ResNet and WideResNet architectures, with extensive configuration options for training and regularization.

W&B project:
https://wandb.ai/matikosowy-none/cifar100-cnn/overview

## Project Structure
```sh
Project Root
├── .gitignore 
├── checkpoints/ 
│ ├── resnet18/ 
│ ├── resnet50/ 
├── cifar100cnn/ 
│ ├── __init__.py 
│ ├── data.py 
│ ├── models.py 
│ ├── train.py 
├── class_names.txt 
├── data/ 
├── LICENSE 
├── main.py 
├── notebooks/ 
│ ├── dataset_analysis.ipynb 
├── README.md 
├── wandb/
├── requirements.txt
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- wandb
- tqdm

Install required packages:
```sh
pip install -r requirements.txt
```

# Usage

Run the script with command line arguments for different operations:

## Training
Train a ResNet-18 from scratch with data augmentation:
```sh
python main.py --mode train --model resnet --version 18 \
              --epochs 100 --lr 0.1 --wd 5e-4 --augment \
              --scheduler cos --ls 0.1 --mixup
```

Fine-tune a pretrained ResNet-50 (unfreezing 2 layers):
```sh
python main.py --mode train --model resnet --version 50 \
              --pretrained --unfreeze 2 --classes 100 \
              --epochs 50 --lr 0.01 --augment
```

Train a WideResNet:
```sh
python main.py --mode train --model wideresnet --classes 50 \
              --epochs 200 --lr 0.1 --mixup --augment
```

## Inference
Evaluate a trained model:
```sh
python main.py --mode inference --path checkpoints/resnet18/fine-tuned/best_model.pth
```

## Key Arguments

### Main Parameters
- `--mode`: Operation mode (`train`/`inference`)
- `--model`: Architecture choice (`resnet`/`wideresnet`)
- `--path`: Model checkpoint path (required for inference)

### Model Configuration
- `--version`: ResNet version (18/50)
- `--pretrained`: Use pretrained ResNet weights
- `--unfreeze`: Number of layers to unfreeze (0-5)
- `--classes`: Number of output classes (1-100)

### Training Parameters
- `--epochs`: Number of training epochs
- `--lr`: Initial learning rate
- `--wd`: Weight decay (L2 regularization)
- `--scheduler`: LR scheduler (`adam`, `sgd`, `cos`, `reduce`, `1cycle`)

### Regularization
- `--augment`: Enable standard data augmentation
- `--mixup`: Enable MixUp augmentation
- `--ls`: Label smoothing factor (0.0-1.0)

### Checkpoint Handling
- `--resume`: Resume training from last checkpoint

## Data Loading
The data pipeline supports:
- Automatic download and caching of CIFAR-100
- Customizable number of classes (1-100)
- Data augmentation (crops, flips, etc.) with `--augment`
- Normalization and standardization

## Models

### ResNet
- Versions: 18/50
- Pretrained initialization option
- Gradual unfreezing for fine-tuning

### WideResNet
- Default configuration: 28 layers with widen factor 10

## Logging and Monitoring
Training metrics are tracked with Weights & Biases:
- Loss curves and accuracy metrics
- Hyperparameter tracking
- System resource monitoring

Configure your W&B account and project name in `TrainerConfig` before running.

## Checkpointing
Automatic checkpoint features:
- Best model saving (validation accuracy)
- Last checkpoint preservation
- Training resume capability
- Model config and metadata storage