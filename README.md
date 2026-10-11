# Deep Learning Assignment 02

**Student:** Naina Awan  
**Roll Number:** 23F-0777  
**Batch:** 23

## Project Overview

This assignment investigates how a small convolutional neural network performs when training data is limited and test images are altered. It compares a baseline model with different data-augmentation and regularization strategies, then evaluates the final model on clean and shifted CIFAR-10 images.

## Dataset

- **Dataset:** CIFAR-10
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck
- **Training subset:** 500 images selected from the CIFAR-10 training set, with 50 images per class
- **Train/validation split:** 400 training images and 100 validation images
- **Test sets:** clean CIFAR-10 test images, synthetically shifted images, and images with hidden corruption

The shifted test set includes transformations such as rotation, brightness/contrast changes, blur, noise, and erased image regions.

## Methodology

The notebook covers:

- Reproducible experiments using fixed random seeds
- A baseline model trained without data augmentation
- Geometric and photometric augmentation experiments
- Mixup and label smoothing
- Regularization and training strategies, including Batch Normalization, Dropout, learning-rate scheduling, and early stopping
- Ablation studies to compare individual changes
- Error analysis and test-time augmentation (TTA)

The final configuration uses horizontal flipping, geometric and photometric transformations, colour variation, Cutout/Random Erasing, Mixup, and label smoothing. The model has **271,578 trainable parameters** (as reported by the notebook's parameter count).

## Results

Mean accuracy and standard deviation are reported across three runs (seeds 0, 1, and 2).

| Metric                | Baseline: no augmentation |    Final model |
| --------------------- | ------------------------: | -------------: |
| Training accuracy     |           100.00% ± 0.00% | 52.33% ± 4.13% |
| Validation accuracy   |            42.33% ± 1.15% | 32.33% ± 5.51% |
| Clean test accuracy   |            41.43% ± 1.18% | 36.43% ± 2.29% |
| Shifted test accuracy |            14.38% ± 0.84% | 22.65% ± 1.70% |

The final model improved shifted-test accuracy by **8.27 percentage points** compared with the baseline. Its clean-test accuracy was lower, showing a trade-off between performance on clean images and robustness to shifted images. In the ablation study, adding Cutout/Random Erasing produced the largest single improvement in shifted-test accuracy (**+6.10 percentage points** at that stage of the experiment).

## How to Run

1. Open `DL_ASS02_23F_0777.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Ensure Python and the notebook's required libraries are available. The notebook uses TensorFlow/Keras, NumPy, pandas, Matplotlib, SciPy, and scikit-learn.
3. Run the notebook cells from top to bottom. The notebook downloads CIFAR-10 through TensorFlow Keras and creates the small training subset and test variants used in the experiments.

A GPU can speed up training, but availability depends on the environment. Exact results may vary with software versions or hardware.

## Repository Contents

- `DL_ASS02_23F_0777.ipynb` — notebook containing the implementation, experiments, plots, and analysis
- `README.md` — project overview and summary of results

## Conclusion

Training on a very small subset of CIFAR-10 leads to strong overfitting in the baseline model. The experiments show that augmentation and regularization choices affect clean accuracy and robustness differently. The final configuration performs better on shifted images than the baseline, although its clean-test accuracy is lower.
