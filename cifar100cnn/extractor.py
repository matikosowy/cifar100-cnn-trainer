import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import torch
import numpy as np
import torch.nn.functional as F
from cifar100cnn.models import *

class FeatureExtractor:
    def __init__(self, device):
        self.device = device

    def extract_features(self, model, loader):
        features, labels = [], []
        
        model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)

                if isinstance(model, ResNet):
                    output = self._extract_resnet_features(model, images)
                elif isinstance(model, WideResNet):
                    output = self._extract_wide_resnet_features(model, images)

                features.append(output.to(self.device).numpy())
                labels.append(targets.to(self.device).numpy())

        return np.vstack(features), np.hstack(labels)

    def _extract_resnet_features(self, model, x):
        x = model.model.conv1(x)
        x = model.model.bn1(x)
        x = model.model.relu(x)
        x = model.model.maxpool(x)
        x = model.model.layer1(x)
        x = model.model.layer2(x)
        x = model.model.layer3(x)
        x = model.model.layer4(x)
        x = model.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _extract_wide_resnet_features(self, model, x):
        x = model.conv1(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = F.relu(model.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        return x
    
def compute_class_representatives(features, labels, num_samples, save_path=None):
    class_representatives = {}
    unique_classes = np.unique(labels)
    for c in unique_classes:
        class_indices = np.where(labels == c)[0]
        selected_indices = np.random.choice(class_indices, min(num_samples, len(class_indices)), replace=False)
        selected_features = features[selected_indices]

        selected_features = normalize(selected_features, norm='l2', axis=1)

        class_representatives[c] = np.mean(selected_features, axis=0)
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, class_representatives)

    return class_representatives


def classify_knn(test_features, class_representatives, n_neighbors=1):
    train_features = np.vstack(list(class_representatives.values()))
    train_labels = np.array(list(class_representatives.keys()))

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(train_features, train_labels)
    predicted_labels = knn.predict(test_features)

    return predicted_labels

def evaluate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100

def run_knn(models, feature_extractor, train_loader, test_loader, classes, device, stage_number, stage_name, results_dir, samples_per_class):
    """
        If it's possible, we will restore the already extracted features and classified data from cache.
    """
    print(f"\nETAP {stage_number}: Klasyfikacja k-NN:")
    
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        # Cache dla cech testowych
        test_features_path = f"cache/{model_name}_test_features.npy"
        test_labels_path = f"cache/{model_name}_test_labels.npy"
        
        if os.path.exists(test_features_path):
            test_features = np.load(test_features_path)
            test_labels = np.load(test_labels_path)
        else:
            test_features, test_labels = feature_extractor.extract_features(model, test_loader)
            np.save(test_features_path, test_features)
            np.save(test_labels_path, test_labels)

        # Filtracja danych testowych
        mask_test = np.isin(test_labels, classes)
        test_features_filtered, test_labels_filtered = test_features[mask_test], test_labels[mask_test]

        # Przetwarzanie dla różnych rozmiarów podzbiorów
        for num_samples in samples_per_class:
            results_path = f"{results_dir}/{model_name}_{num_samples}_class.npy"
            
            if os.path.exists(results_path):
                predictions = np.load(results_path)
                accuracy = evaluate_accuracy(predictions, test_labels_filtered)
                print(f"\tpodzbiory {num_samples}-elementowe, accuracy = {accuracy:.2f}% (wczytane z pliku)")
            else:
                # Cache dla cech treningowych
                train_features_path = f"cache/{model_name}_train_features.npy"
                train_labels_path = f"cache/{model_name}_train_labels.npy"
                
                if os.path.exists(train_features_path):
                    train_features = np.load(train_features_path)
                    train_labels = np.load(train_labels_path)
                else:
                    train_features, train_labels = feature_extractor.extract_features(model, train_loader)
                    np.save(train_features_path, train_features)
                    np.save(train_labels_path, train_labels)

                # Filtracja danych treningowych
                mask_train = np.isin(train_labels, classes)
                train_features_filtered, train_labels_filtered = train_features[mask_train], train_labels[mask_train]

                # Obliczenia i zapis wyników
                class_representatives = compute_class_representatives(train_features_filtered, train_labels_filtered, num_samples)
                predictions = classify_knn(test_features_filtered, class_representatives)
                accuracy = evaluate_accuracy(predictions, test_labels_filtered)

                os.makedirs(results_dir, exist_ok=True)
                np.save(results_path, predictions)
                print(f"\tpodzbiory {num_samples}-elementowe, accuracy = {accuracy:.2f}% (nowo obliczone)")