# VisionNet: Comparing VGG-16 and ResNet-18 from Scratch

This project implements and compares two foundational convolutional neural network (CNN) architectures — **VGG-16 (Version C)** and **ResNet-18** — on a custom 3-class image classification task. All models are implemented from scratch using PyTorch to explore how residual connections improve training efficiency and generalization.

---

##  Objective

- Train and evaluate deep CNN models without using pretrained architectures.
- Understand the architectural trade-offs between VGG-16 and ResNet-18.
- Apply regularization and tuning techniques to optimize performance.
- Visualize and interpret model behavior using plots, metrics, and error analysis.

---

##  Dataset

- **Size**: 30,000 images (10,000 per class)
- **Classes**: Dogs, Cars, Food
- **Resolution**: 64x64 pixels
- **Note**: The dataset was provided as part of a coursework assignment and is not publicly distributed.

---

##  Architectures

###  VGG-16 (Version C)
- Adapted to 64×64 input (modified from original 224×224 spec).
- Dropout layers and learning rate scheduler.
- Optimizer experiments: SGD, Adam, RMSprop.
- Batch sizes tested: 32, 64, 128.
- Regularization: Early stopping, data augmentation.

###  ResNet-18
- Custom residual blocks with identity mappings.
- Same preprocessing pipeline and data splits as VGG.
- Evaluation and plots mirror those used for VGG-16 to enable direct comparison.

---

##  Evaluation Metrics

- Accuracy (Training / Validation / Test)
- Loss curves across epochs
- Confusion matrix with class-wise analysis
- Precision, Recall, F1-score
- Misclassified image visualization
- TensorBoard / WandB logging (optional)
- Weight saving for best performing model

---

##  Results Summary

| Metric        | VGG-16       | ResNet-18    |
|---------------|--------------|--------------|
| Test Accuracy | ~80%         | ~85%         |
| Convergence   | Slower       | Faster       |
| Generalization| Moderate     | Strong       |
| Overfitting   | Noticeable   | Controlled   |

---

##  Visual Results


###  Best VGG Model — Accuracy & Loss Curves

![Best VGG model's accuracy and loss over epochs graphs](https://github.com/user-attachments/assets/d62184b1-e502-46c3-9be4-acf874ca7365)

---

###  Best VGG Model — Confusion Matrix

![Best VGG Model's Confusion matrix](https://github.com/user-attachments/assets/f81b1622-de86-4670-a662-56d5426dd6c9)

---

###  Best ResNet-18 — Accuracy & Loss Curves

![Best Resnet18 model's accuracy and loss over epochs graphs](https://github.com/user-attachments/assets/01a7e0fa-facd-474a-bf21-76b6b8c53ba8)

---

###  Best ResNet-18 — Confusion Matrix

![Best Resnet18 Model's Confusion matrix](https://github.com/user-attachments/assets/d32626b9-162c-47f4-8e92-9a999290a650)

---

###  VGG vs ResNet18 — Training Performance Comparison

![Graph comparison between vgg and resnet18](https://github.com/user-attachments/assets/93b54fe4-0cc5-4cf2-b947-afc923d5ff82)

---

##  Key Takeaways

- **Residual connections** in ResNet greatly mitigate the vanishing gradient problem and improve learning in deeper networks.
- Regularization and optimizer choice significantly impact model performance.
- Misclassification analysis reveals patterns that inform future model improvements.

---

##  Tools & Libraries

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- TensorBoard (optional)

---

##  References

- [VGG Paper (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
- [ResNet Paper (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)

---
