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

def classify_knn(test_features, class_representatives, n_neighbors=1):
    train_features = np.vstack(list(class_representatives.values()))
    train_labels = np.array(list(class_representatives.keys()))

    #train_features = normalize(train_features, norm='l2', axis=1)
    #test_features = normalize(test_features, norm='l2', axis=1)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(train_features, train_labels)
    predicted_labels = knn.predict(test_features)

    return predicted_labels

def evaluate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100

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

NUM_SAMPLES_PER_CLASS = [1, 5, 10]

def run_stage_2(models, feature_extractor, train_loader, test_loader, classes_A, device):
    print("\nETAP 2: Klasyfikacja k-NN dla klas A (treningowych)")
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        train_features, train_labels = feature_extractor.extract_features(model, train_loader)
        test_features, test_labels = feature_extractor.extract_features(model, test_loader)

        mask_train_A = np.isin(train_labels, classes_A)
        mask_test_A = np.isin(test_labels, classes_A)
        train_features_A, train_labels_A = train_features[mask_train_A], train_labels[mask_train_A]
        test_features_A, test_labels_A = test_features[mask_test_A], test_labels[mask_test_A]

        for num_samples in [1, 5, 10]:
            save_path = f"etap2/{model_name}_{num_samples}.npy"
            class_representatives = compute_class_representatives(train_features_A, train_labels_A, num_samples, save_path)
            
            predictions_A = classify_knn(test_features_A, class_representatives)
            accuracy_A = evaluate_accuracy(predictions_A, test_labels_A)
            print(f"\tpodzbiory {num_samples}-elementowe, accuracy = {accuracy_A:.2f}%")

def run_stage_3(models, feature_extractor, train_loader, test_loader, classes_B, device):
    print("\nETAP 3: Klasyfikacja k-NN dla klas B (niewidziane klasy)")
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        train_features, train_labels = feature_extractor.extract_features(model, train_loader)
        test_features, test_labels = feature_extractor.extract_features(model, test_loader)

        mask_train_B = np.isin(train_labels, classes_B)
        mask_test_B = np.isin(test_labels, classes_B)
        train_features_B, train_labels_B = train_features[mask_train_B], train_labels[mask_train_B]
        test_features_B, test_labels_B = test_features[mask_test_B], test_labels[mask_test_B]

        for num_samples in NUM_SAMPLES_PER_CLASS:
            save_path = f"etap3/{model_name}_{num_samples}.npy"
            class_representatives_B = compute_class_representatives(train_features_B, train_labels_B, num_samples, save_path)
            
            predictions_B = classify_knn(test_features_B, class_representatives_B)
            accuracy_B = evaluate_accuracy(predictions_B, test_labels_B)
            print(f"\tpodzbiory {num_samples}-elementowe, accuracy = {accuracy_B:.2f}%")