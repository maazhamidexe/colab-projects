# Image Segmentation using U-Net and FCN with a VGG16 backbone
---

## 🛠️ Technologies & Libraries Used

[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

This project leverages several Python libraries for deep learning, image processing, and data visualization:

*   **Core & Data Handling:**
    *   `Python 3.x`
    *   `NumPy`: For numerical operations, especially array manipulation.
    *   `PIL (Pillow)`: For image loading and basic manipulation.
    *   `glob`: For finding files matching a specific pattern.
    *   `os`: For interacting with the operating system (e.g., file paths).
*   **Deep Learning Framework:**
    *   `PyTorch`: The primary deep learning library used for:
        *   `torch.nn`: Building neural network layers and models.
        *   `torch.optim`: Implementing optimization algorithms (e.g., Adam).
        *   `torch.utils.data.Dataset` & `DataLoader`: Creating custom datasets and efficient data loading pipelines.
        *   `torchvision.transforms.v2`: Applying image and mask transformations.
        *   `torchvision.models`: Accessing pre-trained models (like VGG16 for FCN).
*   **Metrics & Visualization:**
    *   `torchmetrics`: For calculating standard evaluation metrics (JaccardIndex/mIoU, Accuracy).
    *   `Matplotlib`: For plotting training curves, bar charts, and visualizing images/masks.
    *   `tqdm`: For displaying progress bars during iterative processes like training.
*   **Model Summary (Optional but recommended in notebook):**
    *   `torchsummary`: For generating a textual summary of PyTorch model architectures.

---

## 🚀 Setup and Execution

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Dataset:**
    *   Ensure your dataset is placed in the `Dataset_Final_Exam_CV/cityscapes_data/` directory, with `train` and `val` subfolders.
    *   Each image in these folders should be a `.jpg` (or `.png` - adjust `CityscapesDataset` class if needed) file where the left half is the original image and the right half is the corresponding colored annotation mask.
3.  **Install Dependencies:**
    It's recommended to use a virtual environment (e.g., `venv` or `conda`).
    ```bash
    pip install torch torchvision torchaudio torchmetrics matplotlib numpy Pillow tqdm torchsummary
    # Or if you have a requirements.txt:
    # pip install -r requirements.txt 
    ```
    Ensure you install the correct PyTorch version for your CUDA setup if using a GPU.
4.  **Run the Jupyter Notebook:**
    *   Open and run the `semantic_segmentation_notebook.ipynb` file using Jupyter Lab or Jupyter Notebook.
    *   The notebook will guide you through data loading, model training, evaluation, and visualization.

---

## 📈 Results and Comparison

Both U-Net and FCN-VGG16 models were trained for 3 epochs. The results reflect an early stage of training.

### Training Metrics Summary (Validation Set - Epoch 3)

| Model      | Val Loss | Val mIoU | Val Dice | Val PixAcc |
| :--------- | :------- | :------- | :------- | :--------- |
| U-Net      | 0.4975   | 0.3751   | 0.1048   | 0.9776     |
| FCN-VGG16  | 0.4869   | 0.4941   | 0.2364   | 0.9957     |

### Discussion

*   **Overall Performance:** After 3 epochs, the **FCN-VGG16** model generally outperformed the U-Net model on the validation set, particularly in terms of **mIoU** (0.4941 vs 0.3751) and **Pixel Accuracy** (0.9957 vs 0.9776). This is likely attributable to the FCN leveraging a pre-trained VGG16 encoder, which provides a strong feature extraction baseline. The validation loss was also slightly lower for FCN-VGG16.

*   **U-Net:**
    *   Showed consistent improvement in training and validation loss.
    *   Validation mIoU peaked at Epoch 2 (0.3753) and slightly dipped at Epoch 3 (0.3751), suggesting it might be approaching a local optimum or requires more fine-tuning/epochs.
    *   The Dice coefficient remained relatively low for U-Net, indicating potential struggles with class overlap, especially for smaller or imbalanced classes.

*   **FCN-VGG16:**
    *   Demonstrated faster convergence in mIoU, reaching a significantly higher value than U-Net.
    *   The validation Dice coefficient was higher than U-Net's in Epoch 1 (0.3987) but decreased in subsequent epochs (0.3166, 0.2364). This behavior is somewhat unusual given the mIoU improvement and might indicate that the combined loss (CrossEntropy + Dice) weighting or the Dice loss component itself needs further tuning for this architecture, or that it's struggling with specific class boundaries despite overall better region identification. The training Dice score was also notably low for FCN.

*   **Pixel Accuracy:** Both models achieved very high pixel accuracy. This metric can be misleading in semantic segmentation if large background areas or dominant classes (like road, sky) are easily classified, masking poorer performance on smaller, more critical classes.

*   **Training Curves & Visualizations:**
    *   The training curves (loss, mIoU, Dice vs. epoch) would provide more insight into the learning dynamics. *(Refer to `unet_training.png` and `fcn_training.png` in the `images/` directory)*.
    *   Qualitative inspection of predicted masks *(refer to `predicted_images_of_both.png`)* would reveal how well each model delineates object boundaries and handles different classes. FCN-VGG16 might capture larger semantic regions better initially, while U-Net, with its skip connections, has the potential for finer boundary details with more training.

### Conclusion from Results
The FCN-VGG16, with its pre-trained backbone, showed a clear advantage in learning meaningful representations faster, leading to better mIoU within the limited 3-epoch training. However, the Dice coefficient behavior for FCN suggests potential areas for improvement in loss function design or handling class imbalance. U-Net, while starting slower, demonstrates a steady learning trend and could potentially achieve comparable or better results with more extensive training due to its architecture designed for precise localization.

For both models, 3 epochs are insufficient for full convergence.

---

## 🔮 Future Work & Enhancements

*   **Extended Training:** Train for more epochs with learning rate schedulers and early stopping.
*   **Hyperparameter Tuning:** Optimize learning rates, batch sizes, optimizer parameters, and loss function weights.
*   **Data Augmentation:** Implement more sophisticated data augmentation techniques (e.g., using Albumentations) to improve model robustness and generalization.
*   **Advanced Architectures:** Explore state-of-the-art models like DeepLabV3+, PSPNet, or Transformer-based segmentation models.
*   **Loss Function Refinement:** Experiment with Focal Loss, Lovász-Softmax loss, or adaptive weighting for the combined loss.
*   **Post-processing:** Apply Conditional Random Fields (CRF) to refine segmentation boundaries.
*   **More Extensive Comparison:** Conduct a more thorough evaluation with more epochs and possibly on a test set, including per-class metric analysis.

---

## 🤝 Contribution

Feel free to fork this repository, open issues, or submit pull requests with improvements or new features!

---

Stay cool and keep coding! 😎
---

## 🛠️ Technologies & Libraries Used

This project leverages several Python libraries for deep learning, image processing, and data visualization:

*   **Core & Data Handling:**
    *   `Python 3.x`
    *   `NumPy`: For numerical operations, especially array manipulation.
    *   `PIL (Pillow)`: For image loading and basic manipulation.
    *   `glob`: For finding files matching a specific pattern.
    *   `os`: For interacting with the operating system (e.g., file paths).
*   **Deep Learning Framework:**
    *   `PyTorch`: The primary deep learning library used for:
        *   `torch.nn`: Building neural network layers and models.
        *   `torch.optim`: Implementing optimization algorithms (e.g., Adam).
        *   `torch.utils.data.Dataset` & `DataLoader`: Creating custom datasets and efficient data loading pipelines.
        *   `torchvision.transforms.v2`: Applying image and mask transformations.
        *   `torchvision.models`: Accessing pre-trained models (like VGG16 for FCN).
*   **Metrics & Visualization:**
    *   `torchmetrics`: For calculating standard evaluation metrics (JaccardIndex/mIoU, Accuracy).
    *   `Matplotlib`: For plotting training curves, bar charts, and visualizing images/masks.
    *   `tqdm`: For displaying progress bars during iterative processes like training.
*   **Model Summary (Optional but recommended in notebook):**
    *   `torchsummary`: For generating a textual summary of PyTorch model architectures.

---

## 🚀 Setup and Execution

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
2.  **Dataset:**
    *   Ensure your dataset is placed in the `Dataset_Final_Exam_CV/cityscapes_data/` directory, with `train` and `val` subfolders.
    *   Each image in these folders should be a `.jpg` (or `.png` - adjust `CityscapesDataset` class if needed) file where the left half is the original image and the right half is the corresponding colored annotation mask.
3.  **Install Dependencies:**
    It's recommended to use a virtual environment (e.g., `venv` or `conda`).
    ```bash
    pip install torch torchvision torchaudio torchmetrics matplotlib numpy Pillow tqdm torchsummary
    # Or if you have a requirements.txt:
    # pip install -r requirements.txt 
    ```
    Ensure you install the correct PyTorch version for your CUDA setup if using a GPU.
4.  **Run the Jupyter Notebook:**
    *   Open and run the `semantic_segmentation_notebook.ipynb` file using Jupyter Lab or Jupyter Notebook.
    *   The notebook will guide you through data loading, model training, evaluation, and visualization.

---

## 📈 Results and Comparison

Both U-Net and FCN-VGG16 models were trained for 3 epochs. The results reflect an early stage of training.

### Training Metrics Summary (Validation Set - Epoch 3)

| Model      | Val Loss | Val mIoU | Val Dice | Val PixAcc |
| :--------- | :------- | :------- | :------- | :--------- |
| U-Net      | 0.4975   | 0.3751   | 0.1048   | 0.9776     |
| FCN-VGG16  | 0.4869   | 0.4941   | 0.2364   | 0.9957     |

### Discussion

*   **Overall Performance:** After 3 epochs, the **FCN-VGG16** model generally outperformed the U-Net model on the validation set, particularly in terms of **mIoU** (0.4941 vs 0.3751) and **Pixel Accuracy** (0.9957 vs 0.9776). This is likely attributable to the FCN leveraging a pre-trained VGG16 encoder, which provides a strong feature extraction baseline. The validation loss was also slightly lower for FCN-VGG16.

*   **U-Net:**
    *   Showed consistent improvement in training and validation loss.
    *   Validation mIoU peaked at Epoch 2 (0.3753) and slightly dipped at Epoch 3 (0.3751), suggesting it might be approaching a local optimum or requires more fine-tuning/epochs.
    *   The Dice coefficient remained relatively low for U-Net, indicating potential struggles with class overlap, especially for smaller or imbalanced classes.

*   **FCN-VGG16:**
    *   Demonstrated faster convergence in mIoU, reaching a significantly higher value than U-Net.
    *   The validation Dice coefficient was higher than U-Net's in Epoch 1 (0.3987) but decreased in subsequent epochs (0.3166, 0.2364). This behavior is somewhat unusual given the mIoU improvement and might indicate that the combined loss (CrossEntropy + Dice) weighting or the Dice loss component itself needs further tuning for this architecture, or that it's struggling with specific class boundaries despite overall better region identification. The training Dice score was also notably low for FCN.

*   **Pixel Accuracy:** Both models achieved very high pixel accuracy. This metric can be misleading in semantic segmentation if large background areas or dominant classes (like road, sky) are easily classified, masking poorer performance on smaller, more critical classes.

*   **Training Curves & Visualizations:**
    *   The training curves (loss, mIoU, Dice vs. epoch) would provide more insight into the learning dynamics. *(Refer to `unet_training.png` and `fcn_training.png` in the `images/` directory)*.
    *   Qualitative inspection of predicted masks *(refer to `predicted_images_of_both.png`)* would reveal how well each model delineates object boundaries and handles different classes. FCN-VGG16 might capture larger semantic regions better initially, while U-Net, with its skip connections, has the potential for finer boundary details with more training.

### Conclusion from Results
The FCN-VGG16, with its pre-trained backbone, showed a clear advantage in learning meaningful representations faster, leading to better mIoU within the limited 3-epoch training. However, the Dice coefficient behavior for FCN suggests potential areas for improvement in loss function design or handling class imbalance. U-Net, while starting slower, demonstrates a steady learning trend and could potentially achieve comparable or better results with more extensive training due to its architecture designed for precise localization.

For both models, 3 epochs are insufficient for full convergence.

---

## 🔮 Future Work & Enhancements

*   **Extended Training:** Train for more epochs with learning rate schedulers and early stopping.
*   **Hyperparameter Tuning:** Optimize learning rates, batch sizes, optimizer parameters, and loss function weights.
*   **Data Augmentation:** Implement more sophisticated data augmentation techniques (e.g., using Albumentations) to improve model robustness and generalization.
*   **Advanced Architectures:** Explore state-of-the-art models like DeepLabV3+, PSPNet, or Transformer-based segmentation models.
*   **Loss Function Refinement:** Experiment with Focal Loss, Lovász-Softmax loss, or adaptive weighting for the combined loss.
*   **Post-processing:** Apply Conditional Random Fields (CRF) to refine segmentation boundaries.
*   **More Extensive Comparison:** Conduct a more thorough evaluation with more epochs and possibly on a test set, including per-class metric analysis.

---

## 🤝 Contribution

Feel free to fork this repository, open issues, or submit pull requests with improvements or new features!

---

Stay cool and keep coding! 😎
