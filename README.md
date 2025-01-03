# CIFAR-100 CNN Training and Evaluation

Recruitment project for WUT KNSI GOLEM science club.

This repository contains code for training and evaluating Convolutional Neural Networks (CNNs) on the CIFAR-100 dataset. The project includes implementations of ResNet and WideResNet architectures, along with training scripts and data loading utilities.

W&B project:
https://wandb.ai/matikosowy-none/cifar100-cnn?nw=nwusermatikosowy

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

You can install the required packages using pip:

```sh
pip install -r requirements.txt
```

# Usage

## Training
To train a model, run the main.py script with the MODE set to 'train'. You can configure various training parameters in the script.

## Inference
To perform inference using a pre-trained model, set the MODE to 'inference' and specify the path to the model checkpoint in MODEL_FOR_INFERENCE.

## Configuration
The training and model configurations are defined in the TrainerConfig class in train.py. You can customize parameters such as learning rate, number of epochs, optimizer type, and more.

## Data Loading
The CIFAR-100 dataset is loaded and preprocessed using the CIFAR100DataModule class in data.py. This class handles data augmentation, normalization, and splitting into training, validation, and test sets.

## Models
The repository includes implementations of ResNet and WideResNet architectures in models.py. You can choose between these models and their parameters.

## Logging and Visualization
Training progress and metrics are logged using Weights & Biases (wandb). Make sure to set up your wandb account and configure the project name in TrainerConfig.
