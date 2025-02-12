import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import torch
import numpy as np
import torch.nn.functional as F
import wandb
from cifar100cnn.models import *

class FeatureExtractor:
    def __init__(self, device):
        """Init the extractor with the device."""
        self.device = device

    def extract_features(self, model, loader):
        """
            Retrieves the linear layer (features), ommits the classification layer.
            Returns stacked feature vectors and corresponding labels as numpy arrays.

            Args:
                model: trained model to extract features from
                loader: DataLoader to process
        """
        features, labels = [], []
        
        model.eval()
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)

                if isinstance(model, ResNet):
                    output = self._extract_resnet_features(model, images)
                elif isinstance(model, WideResNet):
                    output = self._extract_wide_resnet_features(model, images)

                features.append(output.cpu().numpy())
                labels.append(targets.cpu().numpy())

        return np.vstack(features), np.hstack(labels)

    def _extract_resnet_features(self, model, x):
        """
            Adjusted to the (basic) resnet architecture.
            Forward pass through convolutional blocks and average pooling.
        """
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
        """
            Adjusted to the wide resnet architecture.
            Forward pass through convolutional blocks with custom pooling.
        """
        x = model.conv1(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = F.relu(model.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        return x
    
def compute_class_representatives(features, labels, num_samples, save_path=None):
    """
        Creates vectors for each class by averaging features.
        
        Args:
            features: array of features' vectors
            labels: class labels
            num_samples: num of samples per class
            (optional) save_path: path to save representative
    """
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
    """
        Returns results (array of predicted class labels) of the KNN classification with class representatives.
        Default value of n-neighbors is 1 --- returns the best accuracies for all models.
        
        Args:
            test_features: feature vectors to classify
            class_representatives: dict of class representatives
            n_neighbors: number of neighbors in knn classifier
    """

    train_features = np.vstack(list(class_representatives.values()))
    train_labels = np.array(list(class_representatives.keys()))

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(train_features, train_labels)

    return knn.predict(test_features)

def evaluate_accuracy(predictions, true_labels):
    """
        Returns classification accuracy in percentage.
    """
    return np.mean(predictions == true_labels) * 100

def run_knn(models, feature_extractor, train_loader, test_loader, classes, device, stage_number, stage_name, results_dir, samples_per_class, n_neighbors_list):
    """
        Full evaluation for KNN classification (can be used for BOTH stages).
        Handles: feature extraction, caching, and result logging.

        Args:
            models: models to consider during evaluation
            feature_extractor: given instance of FeatureExtractor
            train_loader: DataLoader of the training part of the dataset
            test_loader: DataLoader of the test part of the dataset
            classes: target classes included in evaluation
            device: comp. device (cpu/gpu)
            stage_number: number of the stage (2/3)
            stage_name: description of the stage
            results_dir: path to the results of the evaluation --- stored in cache
            samples_per_class: list of sample counts per class for creating the representatives
            n_neighbors_list: list of k-values for k-NN evaluation

    """

    print(f"\nETAP {stage_number}: Klasyfikacja k-NN:")
    
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")

        # cache for test features - calculating them each time takes too much time
        test_features_path = f"cache/{model_name}_test_features.npy"
        test_labels_path = f"cache/{model_name}_test_labels.npy"
        
        if os.path.exists(test_features_path):
            test_features = np.load(test_features_path)
            test_labels = np.load(test_labels_path)
        else:
            test_features, test_labels = feature_extractor.extract_features(model, test_loader)
            np.save(test_features_path, test_features)
            np.save(test_labels_path, test_labels)

        # filtering test data to match given classes
        mask_test = np.isin(test_labels, classes)
        test_features_filtered, test_labels_filtered = test_features[mask_test], test_labels[mask_test]

        train_features_path = f"cache/{model_name}_train_features.npy"
        train_labels_path = f"cache/{model_name}_train_labels.npy"
        
        if os.path.exists(train_features_path):
            train_features = np.load(train_features_path)
            train_labels = np.load(train_labels_path)
        else:
            train_features, train_labels = feature_extractor.extract_features(model, train_loader)
            np.save(train_features_path, train_features)
            np.save(train_labels_path, train_labels)

        mask_train = np.isin(train_labels, classes)
        train_features_filtered = train_features[mask_train]
        train_labels_filtered = train_labels[mask_train]

        accuracies_dict = {k: [] for k in n_neighbors_list}
        samples_list = sorted(samples_per_class)

        for num_samples in samples_per_class:
            class_representatives = compute_class_representatives(train_features_filtered, train_labels_filtered, num_samples)
            
            for k in n_neighbors_list:
                results_path = f"{results_dir}/{model_name}_{num_samples}_k{k}_class.npy"
                
                if os.path.exists(results_path):
                    predictions = np.load(results_path)
                    accuracy = evaluate_accuracy(predictions, test_labels_filtered)
                    print(f"\tpodzbiory {num_samples}-elementowe, knn k={k}, accuracy = {accuracy:.2f}% (wczytane z pliku)")
                else:
                    predictions = classify_knn(test_features_filtered, class_representatives, n_neighbors=k)
                    accuracy = evaluate_accuracy(predictions, test_labels_filtered)
                    os.makedirs(results_dir, exist_ok=True)
                    np.save(results_path, predictions)
                    print(f"\tpodzbiory {num_samples}-elementowe, knn k={k}, accuracy = {accuracy:.2f}%")

                accuracies_dict[k].append(accuracy)

                # logging results in wandb
                wandb.log({
                    "stage": stage_number,
                    "model": model_name,
                    "samples_per_class": num_samples,
                    "k_neighbors": k,
                    "accuracy": accuracy
                })

        # plotting for each k-number
        for k in n_neighbors_list:
            table = wandb.Table(
                data=[[str(s), acc] for s, acc in zip(samples_list, accuracies_dict[k])],
                columns=["samples_per_class", "accuracy"]
            )
            
            line_plot = wandb.plot.line(
                table,
                x="samples_per_class",
                y="accuracy",
                title=f"Stage {stage_number} - {model_name} (k={k})"
            )
            
            wandb.log({
                f"Stage {stage_number}/{model_name}/k_{k}/Accuracy": line_plot
            })