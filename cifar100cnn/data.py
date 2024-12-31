import torch
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms


class CIFAR100DataModule:
    """Klasa do ładowania i przetwarzania danych CIFAR-100."""
    def __init__(self, 
                 batch_size= 128,
                 num_workers = 4,
                 num_classes = 100,
                 data_dir = './data',
                 augment = True):
    
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.augment = augment
        
        # Wartości do normalizacji dla CIFAR-100
        self.mean = [0.5071, 0.4867, 0.4408]
        self.std = [0.2675, 0.2565, 0.2761]
        
        self.selected_classes = None
        self.class_names = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _get_transforms(self):
        """Zwraca transformacje dla zbioru treningowego i testowego."""
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(10 if self.augment else 0),
            *([transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2, hue=0.1)]
                if self.augment else [] ),
            transforms.RandomErasing(p=0.1 if self.augment else 0),
            transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]),
            transforms.Normalize(self.mean, self.std)
        ])

        test_transform = transforms.Compose([
            transforms.Compose([
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
            ]),
            transforms.Normalize(self.mean, self.std)
        ])

        return train_transform, test_transform

    def _select_classes(self, dataset):
        """Zwraca mapowanie klas oryginalnych na wybrane klasy."""
        rng = np.random.RandomState(42)
        self.selected_classes = rng.choice(100, self.num_classes, replace=False)
        self.selected_classes.sort()
        
        class_mapping = {old_idx: new_idx 
                        for new_idx, old_idx in enumerate(self.selected_classes)}
        
        full_class_names = dataset.classes
        self.class_names = [full_class_names[i] for i in self.selected_classes]
        
        return class_mapping

    def _filter_dataset(self, dataset, class_mapping):
        """Filtruje zbiór danych do wybranych klas."""
        indices = [i for i in range(len(dataset)) 
                  if dataset.targets[i] in self.selected_classes]
        
        filtered_dataset = torch.utils.data.Subset(dataset, indices)
        
        filtered_dataset.dataset.targets = [
            class_mapping[target] if target in class_mapping else target 
            for target in filtered_dataset.dataset.targets
        ]
        
        return filtered_dataset

    def setup(self):
        """Pobiera, przetwarza i dzieli zbiór danych na zbiór treningowy, walidacyjny i testowy."""
        train_transform, test_transform = self._get_transforms()
        
        train_dataset = datasets.CIFAR100(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        val_dataset = datasets.CIFAR100(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=test_transform
        )
        
        test_dataset = datasets.CIFAR100(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=test_transform
        )
        
        class_mapping = self._select_classes(train_dataset)
        train_dataset = self._filter_dataset(train_dataset, class_mapping)
        val_dataset = self._filter_dataset(val_dataset, class_mapping)
        test_dataset = self._filter_dataset(test_dataset, class_mapping)
        
        generator = torch.Generator().manual_seed(42)
        train_split = int(0.8 * len(train_dataset))
        
        indices = torch.randperm(len(train_dataset), generator=generator)
        train_indices = indices[:train_split]
        val_indices = indices[train_split:]
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        loader_kwargs = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': True if self.num_workers > 0 else False
        }
        
        self.train_loader = DataLoader(
            train_dataset, 
            shuffle=True,
            generator=generator,
            **loader_kwargs
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            shuffle=False,
            **loader_kwargs
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            shuffle=False,
            **loader_kwargs
        )
        
    @property
    def dataloaders(self):
        if not all([self.train_loader, self.val_loader, self.test_loader]):
            raise RuntimeError("Data module not set up. Call setup() first.")
            
        return self.train_loader, self.val_loader, self.test_loader, self.class_names


def get_cifar_data(batch_size=128, num_workers=4, num_classes=100, 
                   data_dir='./data', augment=False):
    """Domyślna funkcja do pobierania gotowych danych CIFAR-100."""
    data_module = CIFAR100DataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        num_classes=num_classes,
        data_dir=data_dir,
        augment=augment,
    )
    
    data_module.setup()
    return data_module.dataloaders