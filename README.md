# Gradient-based Model Shortcut Detection for Time Series Classification

This repository will host the official implementation for the paper:

**Gradient-based Model Shortcut Detection for Time Series Classification**

---

## 📖 Abstract
Deep learning models have attracted significant research attention in time series classification (TSC) over the past two decades. Recently, deep neural networks (DNNs) have surpassed classical distance-based methods and achieved state-of-the-art performance. Despite this success, DNNs are prone to relying on spurious correlations in training data, which can hinder generalization. For example, a model may incorrectly associate the presence of grass with the label "cat" if most cats in the training set appear on grassy backgrounds.  

However, shortcut behaviors of DNNs in time series remain under-explored. Most existing shortcut studies rely on external attributes such as gender or patient groups, instead of focusing on the internal bias behavior in time series models.  

In this work, we take the first step to investigate and establish the point-based shortcut learning behavior in deep learning for time series classification. We further propose a simple detection method based on other-class gradients to detect shortcuts **without relying on test data or clean training classes**. Our experiments on UCR time series datasets validate the effectiveness of the proposed approach.

---

## How to use
Example of how to use our proposed method
'''python
import torch
import torch.nn as nn
from models.models import resnet18
from SAG import load_data, SAG

# Load the GunPoint datasetn from aeon using our personal function
X_train, y_train, X_test, y_test, n_class = load_data("GunPoint")

# First, create the model architecture
model = resnet18(num_classes=n_class, input_size=X_train.shape[1])

# Then load the saved weights
model.load_state_dict(torch.load("resnet18_gunpoint.pth"))

# Create a data loader for SAG (unseen data)
data_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.from_numpy(X_test).float(), 
        torch.from_numpy(y_test).long()
    ), 
    batch_size=32, 
    shuffle=False
)

# Compute SAG scores
score0, score1 = SAG(model, data_loader, y_test, criterion=nn.CrossEntropyLoss())

print(f"SAG Score Class 0: {score0:.4f}")
print(f"SAG Score Class 1: {score1:.4f}")

How to reproduce a result from our paper
'''python

from SAG import ICMLA_result

ICMLA_results("GunPoint)

---
## 📌 Citation
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ibarra2025SAG,
  title={Gradient-based Model Shortcut Detection for Time Series Classification},
  author={ibarra, cantu, kaixiong, zhang},
  booktitle={2025 International Conference on Machine Learning and Applications (ICMLA)},
  year={2025},
  organization={IEEE}
}
