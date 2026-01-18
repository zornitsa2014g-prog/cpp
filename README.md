# 🐕 Dog Breed Identification using CNN

A Convolutional Neural Network (CNN) project that predicts the breed of a dog from an input image.
The model is trained on a labeled dataset and provides confidence scores and Top-5 predictions.

# 📌 Project Overview

The goal of this project is to build an image classification model that:

- takes an image of a dog as input
- predicts the dog’s breed
- displays the confidence of the prediction
- provides the Top-5 most likely breeds
  
This project is implemented using TensorFlow / Keras and trained on a Kaggle dataset.

# 🧠 Model Architecture

Architecture summary:

- Input layer: 224 × 224 × 3 RGB images
- 4 Convolutional layers with ReLU activation
- MaxPooling layers after each convolution
- Flatten layer
- Fully connected Dense layer
- Dropout (0.5) to reduce overfitting
- Output layer with Softmax activation (multi-class classification)

# 📂 Dataset

- Dataset: Dog Breed Identification
- Source: Kaggle
- Images: RGB images of dogs
- Labels: Dog breed for each image

   The dataset is split automatically into:
- 80% training data
- 20% validation data

# ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Kaggle Notebooks

# 🚀 How to Run the Project

This project is designed to run in a **Kaggle Notebook environment**.

1. Open the Kaggle dataset: Dog Breed Identification.
2. Create a new Kaggle Notebook using the dataset.
3. Paste the code from this repository.
4. Run all cells.

