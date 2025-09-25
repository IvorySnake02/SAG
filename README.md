# Gradient-based Model Shortcut Detection for Time Series Classification

This repository will host the official implementation for the paper:

**Gradient-based Model Shortcut Detection for Time Series Classification**

---

## 📖 Abstract
Deep learning models have attracted significant research attention in time series classification (TSC) over the past two decades. Recently, deep neural networks (DNNs) have surpassed classical distance-based methods and achieved state-of-the-art performance. Despite this success, DNNs are prone to relying on spurious correlations in training data, which can hinder generalization. For example, a model may incorrectly associate the presence of grass with the label "cat" if most cats in the training set appear on grassy backgrounds.  

However, shortcut behaviors of DNNs in time series remain under-explored. Most existing shortcut studies rely on external attributes such as gender or patient groups, instead of focusing on the internal bias behavior in time series models.  

In this work, we take the first step to investigate and establish the point-based shortcut learning behavior in deep learning for time series classification. We further propose a simple detection method based on other-class gradients to detect shortcuts **without relying on test data or clean training classes**. Our experiments on UCR time series datasets validate the effectiveness of the proposed approach.

---

## 🚧 Code Availability
The code for this project is **coming soon**.  
Please stay tuned!  

---

## 📌 Citation
If you find this work useful, please cite our paper:

```bibtex
@article{ibarra2025gradient,
  title={Gradient-based Model Shortcut Detection for Time Series Classification},
  author={Ibarra, Salomon and Cantu, Frida and Zhou, Kaixiong and Zhang, Li},
  year={2025}
}
