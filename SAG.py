import os
import math
import random
import copy
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor

from sklearn.utils import shuffle
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, f1_score,
    average_precision_score, precision_recall_curve, roc_auc_score,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import permutations
from collections import OrderedDict
from aeon.datasets import load_classification
from models.models import resnet18
from tabulate import tabulate


"""# Functions"""

# data cleaning
def class_str2num(y):
    unique_labels = set(y)
    # Create an array of zeros with the same size as y
    label = np.zeros(len(y)).astype('float')
    t = 0
    for x in unique_labels:
        ind = y == x
        label[ind]= t
        t = t+1
    return label

def load_data(data_name):

    X_train, y_train = load_classification(data_name, split='train')
    X_test, y_test = load_classification(data_name, split='test')

    y_train = class_str2num(y_train)
    y_test = class_str2num(y_test)

    n_class = len(np.unique(y_train))


    return X_train, y_train, X_test, y_test, n_class

def train_resnet_model(model_resnet, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=1024, verbose=False):
    import warnings
    
    # Suppress the pin_memory warning
    warnings.filterwarnings('ignore', message="'pin_memory' argument is set as true but no accelerator is found")
    
    # Define the optimizer and loss function
    optimizer = optim.Adam(model_resnet.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_resnet.to(device)
    
    # Convert data to PyTorch tensors (keep on CPU initially)
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Optimized DataLoaders with additional parameters
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,           # Parallel data loading
        pin_memory=True,         # Faster CPU-to-GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,           # Fewer workers for testing
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize tracking lists
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model_resnet.train()
        running_loss = 0.0
        correct_train_predictions = 0
        total_train_predictions = 0
        
        for i, data in enumerate(train_loader):
            # Move data to device here (not in advance)
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs, logit = model_resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train_predictions += labels.size(0)
            correct_train_predictions += (predicted == labels).sum().item()
        
        # Testing
        model_resnet.eval()
        test_loss = 0.0
        correct_test_predictions = 0
        total_test_predictions = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move data to device
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                outputs, _ = model_resnet(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test_predictions += labels.size(0)
                correct_test_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = correct_train_predictions / total_train_predictions
        epoch_test_loss = test_loss / len(test_loader)
        test_accuracy = correct_test_predictions / total_test_predictions
        
        # Store metrics
        train_losses.append(epoch_loss)
        test_losses.append(epoch_test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 and verbose==True:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Test Loss: {epoch_test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
    
    if verbose == True:              
        print(f'\nFinal Results:')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}')
        print(f'Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}')
    
    # Cleanup GPU memory
    del train_dataset, train_loader, test_dataset, test_loader
    del optimizer, criterion
    model_resnet.cpu()
    del model_resnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        
def extract_grad(model, dataloader, criterion):
    """
    Extracts gradients from a pre-trained model for a given dataset.

    Args:
        model: The pre-trained PyTorch model.
        dataloader: The DataLoader for the dataset.
        criterion: The loss function used for training.

    Returns:
        A numpy array containing the gradients of the input data with respect to the loss.
    """
    model.eval()
    all_gradients = []
    device = next(model.parameters()).device # Get model device

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True

        model.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        batch_grad = inputs.grad.detach().cpu().numpy()
        all_gradients.append(batch_grad)

    input_grad = np.concatenate(all_gradients, axis=0)

    # Cleanup GPU memory if needed
    del inputs, labels, outputs, loss
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return input_grad

# computes the absolute mean gradient needed by the score function
def compute_abs_grad(grad_data, y_train, target_class=0):
    class_indices = y_train == target_class
    grad_data = grad_data[class_indices,0,:]
    return np.abs(grad_data.mean(axis=0))

# computes the absolute mean gradient needed by the score function for both classes
def compute_perclass_grad(grad_data, y_train):
    grad_class0 = compute_abs_grad(grad_data, y_train, target_class=0)
    grad_class1 =  compute_abs_grad(grad_data, y_train, target_class=1)
    return grad_class0, grad_class1

# computes the class score of a model
def compute_score(a):
    a_min = a.min()
    a_max = a.max()
    a_normalized = (a - a_min) / (a_max - a_min)
    score = a_normalized/a_normalized.sum()
    return score.max()

# computes the per class score of a model
def compute_perclass_score(grad_data, y_train):
  grad_class0, grad_class1 = compute_perclass_grad(grad_data, y_train)
  score0 = compute_score(grad_class0)
  score1 = compute_score(grad_class1)
  return score0, score1

ucr_datasets = [
    "BeetleFly",
    "BirdChicken",
    "Coffee",
    "DistalPhalanxOutlineCorrect",
    "DodgerLoopWeekend",
    "ECG200",
    "ECGFiveDays",
    "Epilepsy2",
    "FreezerRegularTrain",
    "GunPoint",
    "Ham",
    "ItalyPowerDemand",
    "Lightning2",
    "MiddlePhalanxOutlineCorrect",
    "MoteStrain",
    "PowerCons",
    "SharePriceIncrease",
    "SonyAIBORobotSurface2",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "TwoLeadECG",
    "Wafer",
    "WormsTwoClass",
    "Yoga"
]

def set_seed(seed=42):
    # Set seed for reproducibility across numpy, random, and torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# SAG
def SAG(model, data_loader, y_train, criterion):
  """
  input:  model, data_loader, y_train, criterion
  output: SAG score for both classes
  """

  input_grad = extract_grad(model, data_loader, criterion)
  score0, score1 = compute_perclass_score(input_grad, y_train)
  return score0, score1


def plot_score(model, data_loader, y_train, criterion):
  input_grad = extract_grad(model, data_loader, criterion)

  fig, axes = plt.subplots(1, 2, figsize=(12, 6))

  grad_class0, grad_class1 = compute_perclass_grad(input_grad, y_train)

  # Determine the common y-axis limits
  y_min = min(grad_class0.min(), grad_class1.min())
  y_max = max(grad_class0.max(), grad_class1.max())

  axes[0].plot(grad_class0)
  axes[0].set_ylim(y_min, y_max)
  axes[0].set_title('Class 0')

  axes[1].plot(grad_class1)
  axes[1].set_ylim(y_min, y_max)
  axes[1].set_title('Class 1')

  plt.tight_layout()
  plt.legend()
  plt.show()


def display_results_table(results_ICMLA):
    """
    Displays the results from the ICMLA_results function in a formatted table.

    Args:
        results_ICMLA: A list or numpy array containing the results
                       [score0, score1, score_2, score_3].
    """
    headers = ["Metric", "Value"]
    table_data = [
        ["Score Class 0 (Original)", results_ICMLA[0]],
        ["Score Class 1 (Original)", results_ICMLA[1]],
        ["Score Class 0 (Spiked)", results_ICMLA[2]],
        ["Score Class 1 (Spiked)", results_ICMLA[3]]
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


# Reproduces the results for a single dataset from the ICMLA paper.
# This function runs 3 times and takes the avg with different seeds for reproducibility.
def ICMLA_results(dataset):
    """
      args:
            dataset-the name of any dataset of all datasetes used in the ICMLA table

      output:
            final: avg result for the 3 runs of the same dataset (regular & shortcut)
    """
    seeds = [random.randint(1, 1000) for _ in range(3)]


    if dataset not in ucr_datasets:
      raise ValueError(f"Dataset '{dataset}' is not part of the ICMLA results table.")

    results = []
    print(f"Processing dataset: {dataset}")

    X_train, y_train, X_test, y_test, n_class = load_data(dataset)

    spiked_X = X_train.copy()

    class_1 = np.where(y_train == 1)[0]

    spiked_X[class_1, : , 0] = 2


    for seed in seeds:
        set_seed(seed)
        model_resnet = resnet18(num_classes=n_class, input_size=X_train.shape[1])
        train_resnet_model(model_resnet, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=32)
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()), batch_size=32, shuffle=True)
        score0, score1 = SAG(model_resnet, data_loader, y_train, criterion=nn.CrossEntropyLoss())

        spiked_model = resnet18(num_classes=n_class, input_size=X_train.shape[1])
        train_resnet_model(spiked_model, spiked_X, y_train, X_test, y_test, num_epochs=100, batch_size=32)
        data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(spiked_X).float(), torch.from_numpy(y_train).long()), batch_size=32, shuffle=True)
        score_2, score_3 = SAG(spiked_model, data_loader, y_train, criterion=nn.CrossEntropyLoss())
        results.append([score0, score1, score_2, score_3])

    final = np.mean(results, axis=0)



    display_results_table(final)

    return final

