# DevNull-Rayan

This document outlines my submissions in the Rayan AI-Trustworthy Contest ([ai.rayan.global](https://ai.rayan.global)) during the selection phase. I ranked **11th** in this phase. The full problem statements and explorations are available in the corresponding directory.

## Problem 1 - Ducks

We were provided with a dataset of 1,600 images of similar birds, which were noisy, and only 10% of them were labeled.

### Approach
- **Data Augmentation:** Applied proper data augmentation techniques to enhance the dataset.
- **Gradual Learning:** Utilized a method called gradual learning, where the model is initially trained with limited data. Subsequently, data samples that the model is confident about are added to the training set.
- **Model Architecture:** Employed EfficientNet as the neural network architecture.

### Results
- **Accuracy:** Achieved **78.5%** accuracy.
- **Ranking:** Placed **5th** among all submissions.

## Problem 2 - Teeth Caries

The task was to perform segmentation of tooth caries, which was challenged by poisoned data. A significant portion of the data appeared as random noise.

### Approach
- **Noise Reduction:** Removed noisy data using an image entropy technique.
- **Model Architecture:** Used U-Net as the segmentation model.

### Results
- **Dice Score:** Achieved a **0.5110** Dice score.
- **Ranking:** Placed **10th** among all submissions.

## Problem 3 - Taxonomy

This was a hierarchical clustering problem that required developing a classification system for the second layer of the hierarchy. The error could be computed at any arbitrary layer, with the minimum error being selected.

### Approach
- **Data Augmentation:** Implemented smart augmentation techniques to handle imbalanced data due to time constraints.
- **Model Strategy:** Focused on optimizing classification for hierarchical layers.

### Results
- **Average Accuracy:** Achieved a **44.34%** average accuracy.
- **Ranking:** Placed **15th** among all submissions.

---

