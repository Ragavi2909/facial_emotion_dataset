# Facial Emotion Recognition Using CNN

A deep learning project that recognizes human facial emotions from grayscale facial images using a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras.

## 📌 Project Overview

This project implements a CNN-based facial emotion recognition system. The model takes a facial image as input, processes it through multiple convolution and pooling layers, and predicts the corresponding emotion.

The images are resized to **48 × 48 pixels** and converted to grayscale before being given to the CNN.

## 🎯 Objectives

* Detect facial emotions using deep learning.
* Build a CNN model using TensorFlow/Keras.
* Classify facial images into seven different emotions.
* Evaluate the model using accuracy and loss.
* Generate a classification report and confusion matrix.
* Predict the emotion of a newly uploaded image.
* Display prediction confidence for each emotion.

## 😊 Emotion Classes

The model recognizes **7 emotions**:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## 🗂️ Dataset Structure

The dataset is organized into training and testing folders, with separate subfolders for each emotion.

```text
Facial emotion dataset/
│
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

The notebook loads the images using TensorFlow's `image_dataset_from_directory()` function.

## 🧠 CNN Architecture

The CNN consists of three convolutional blocks followed by fully connected layers.

```text
Input Image
48 × 48 × 1
     ↓
Conv2D (32 filters)
     ↓
MaxPooling2D
     ↓
Conv2D (64 filters)
     ↓
MaxPooling2D
     ↓
Conv2D (128 filters)
     ↓
MaxPooling2D
     ↓
Flatten
     ↓
Dense (128)
     ↓
Dropout (0.5)
     ↓
Dense (7)
     ↓
Softmax
     ↓
Predicted Emotion
```

The convolutional layers extract facial features, pooling reduces the spatial dimensions, the dense layer learns high-level representations, and the final softmax layer predicts one of the seven emotions.

## ⚙️ Preprocessing

The following preprocessing steps are used:

* Image resizing to **48 × 48**
* Conversion to **grayscale**
* Categorical labels
* Pixel normalization from `[0, 255]` to `[0, 1]`
* Dataset prefetching using TensorFlow `AUTOTUNE`

The notebook uses a batch size of **64**.

## 🔧 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

## 🚀 Training

The dataset is divided into:

* **80% Training**
* **20% Validation**

The CNN is trained for **20 epochs** using the Adam optimizer and categorical cross-entropy loss.

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

## 📊 Model Evaluation

The model is evaluated on the test dataset using:

### Test Accuracy

```python
test_loss, test_accuracy = model.evaluate(test_dataset)
```

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

for each emotion class.

### Confusion Matrix

A confusion matrix is generated to visualize the relationship between the actual and predicted emotions.

## 📈 Training Visualization

The project plots:

* Training Accuracy
* Validation Accuracy
* Training Loss
* Validation Loss

These plots help analyze the model's learning behavior during training.

## 🔍 Prediction on New Images

The notebook also supports uploading a new facial image and predicting its emotion.

The image is:

1. Uploaded by the user
2. Resized to 48 × 48
3. Converted to grayscale
4. Normalized
5. Passed through the trained CNN
6. Classified into one of the seven emotions

The predicted emotion and confidence score are displayed.

The probability of every emotion is also displayed as a bar chart.

## 💾 Model

The trained model is saved in Keras format:

```text
facial_emotion_cnn.keras
```

The notebook saves the trained model using:

```python
model.save("/content/facial_emotion_cnn.keras")
```

## 📁 Project Structure

A recommended GitHub structure is:

```text
Facial-Emotion-Recognition/
│
├── facial_emotion_detection.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── facial_emotion_cnn.keras
```

> The dataset is not included in this repository because image datasets can be very large.

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Facial-Emotion-Recognition.git
```

### 2. Open the notebook

Open:

```text
facial_emotion_detection.ipynb
```

using Google Colab or Jupyter Notebook.

### 3. Add the dataset

Place the facial emotion dataset in your Google Drive and update the dataset path in the notebook.

```python
source = "/content/drive/MyDrive/Facial emotion dataset"
```

### 4. Run the notebook

Run the cells sequentially to:

* Load the dataset
* Preprocess the images
* Build the CNN
* Train the model
* Evaluate the model
* Generate the confusion matrix
* Test predictions
* Predict emotions from new images

## 📌 Results

The notebook reports the following evaluation metrics:

* Test Accuracy
* Test Loss
* Precision
* Recall
* F1-score
* Confusion Matrix

The actual final accuracy is generated when the notebook is executed on the dataset.

## 🔮 Future Improvements

Possible improvements include:

* Data augmentation
* Class balancing
* Batch normalization
* Learning-rate scheduling
* Transfer learning
* Deeper CNN architectures
* Real-time webcam emotion detection
* Improved performance on minority emotion classes

