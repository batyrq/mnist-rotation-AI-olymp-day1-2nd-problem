# MNIST Rotation Angle Prediction

-----

This repository contains the Jupyter Notebook `mnist-rotation.ipynb`, which provides a solution for predicting the rotation angle of MNIST digits. This was the second problem of the Republican AI Olympiad's first tour.

## Competition Details

  * **Kaggle Competition Link:** [MNIST Rotation](https://www.kaggle.com/competitions/mnist-rotation/overview)
  * **Evaluation Metric:** Accuracy
  * **Achieved Score:** 0.903

## Project Overview

The goal of this project is to predict the rotation angle (from -120 to 120 degrees, with 30-degree increments) of handwritten digits from the MNIST dataset. The approach involves data augmentation, a custom Convolutional Neural Network (CNN) architecture, and training on the augmented data.

The key steps in the solution are:

1.  **Data Loading and Initial Exploration:**
      * Loading the training and testing datasets from pickle files (`train.pkl`, `test.pkl`).
2.  **Data Augmentation:**
      * **Rotation Function:** A `rotate` function is implemented using `scipy.ndimage.rotate` to rotate images by a specified angle while maintaining dimensions and filling empty areas with black.
      * **Noise Generator Class (`NoizeGenerator`):**
          * Applies discrete noise (random pixels replaced with values from a Beta distribution).
          * Applies Gaussian noise.
          * Applies random 1-pixel shifts (up, down, left, right).
          * This class is used to augment the training dataset by adding various types of noise and shifts to rotated images.
      * The original training data is augmented by rotating each image by all allowed angles (`-120, -90, ..., 120`) and then applying the defined noise transformations. This significantly increases the training dataset size and helps the model generalize to rotated and noisy inputs.
3.  **PyTorch Dataset and DataLoader:**
      * **`CustomDataset`:** A custom PyTorch `Dataset` is created for the augmented training data, returning the image, its original digit label, and its rotation angle as a class.
      * **`CustomDataset1`:** A similar custom `Dataset` for the test data, returning images and their digit labels.
      * `DataLoader` is used to efficiently load data in batches for training and inference.
4.  **Angle to Class Mapping:** The unique rotation angles are mapped to integer classes for use with `CrossEntropyLoss`.
5.  **Model Architecture (`CNN`):**
      * A custom CNN model is defined using `torch.nn.Module`.
      * It consists of several convolutional layers followed by batch normalization, ReLU activations, and max-pooling layers to extract features from the input images.
      * A flattening layer and a fully connected layer (`image_fc`) process the image features.
      * Crucially, an `nn.Embedding` layer is used to embed the *digit label* (0-9) into a dense vector. This embedded label is then concatenated with the image features. This allows the model to leverage information about the actual digit to better predict its rotation.
      * A final set of fully connected layers predicts the angle class.
6.  **Training:**
      * The model is trained using `CrossEntropyLoss` and the `Adam` optimizer.
      * Gradient clipping (`torch.nn.utils.clip_grad_norm_`) is applied to prevent exploding gradients.
      * The training loop iterates for 10 epochs, with loss reported per epoch.
7.  **Inference and Submission:**
      * The trained model is set to evaluation mode (`model.eval()`).
      * Predictions are made on the test dataset.
      * The predicted angle classes are mapped back to their original angle values.
      * A `submission.csv` file is generated in the format required by the Kaggle competition.

## Setup and Running the Notebook

To run this notebook, you'll need a Kaggle environment or a local setup with the necessary libraries.

### Prerequisites

  * Python 3.x
  * `pandas`
  * `numpy`
  * `matplotlib`
  * `scipy`
  * `torch`
  * `Pillow` (often installed with torchvision or as a dependency)

### Installation

You can install the required Python packages using pip:

```bash
pip install pandas numpy matplotlib scipy torch
```

### Running the Notebook

1.  **Download the data:** Download the `train.pkl`, `test.pkl`, and `sample_submission.csv` files from the Kaggle competition page and place them in your input directory (e.g., `/kaggle/input/mnist-rotation/` if on Kaggle).
2.  **Open the notebook:** Open `mnist-rotation.ipynb` in a Jupyter environment (Jupyter Lab, Jupyter Notebook, Google Colab, or Kaggle Notebooks).
3.  **Run all cells:** Execute all cells in the notebook sequentially. The script will perform data loading, augmentation, model training, and generate the `submission.csv` file.

-----
