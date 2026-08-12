# 🎙️ Emotion Recognition from Speech Using MFCC and CNN

## 📌 Project Overview

This project is a **Speech Emotion Recognition system** that uses Machine Learning and Deep Learning techniques to identify emotions from human speech.

The system extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features from speech audio and uses a **Convolutional Neural Network (CNN)** to classify the speech into different emotion categories.

---

## 🎯 Objective

The main objective of this project is to develop a Deep Learning model that can automatically recognize human emotions from speech audio.

The project focuses on:

- 🎙️ Processing speech audio files
- 🎧 Extracting MFCC features
- 🔄 Applying audio data augmentation
- 🧠 Training a CNN model
- 📊 Evaluating model performance
- 🎯 Predicting emotions from new audio files

---

## 📂 Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

- 🎵 **Dataset:** RAVDESS
- 🎧 **Total Audio Files:** 1440
- 🎭 **Number of Emotion Classes:** 8

### 😊 Emotion Classes

1. 😐 Neutral
2. 😌 Calm
3. 😄 Happy
4. 😢 Sad
5. 😠 Angry
6. 😨 Fearful
7. 🤢 Disgust
8. 😲 Surprised

---

## 🛠️ Technologies Used

- 🐍 Python
- ☁️ Google Colab
- 🤖 TensorFlow
- 🧠 Keras
- 🎵 Librosa
- 🔢 NumPy
- 🐼 Pandas
- 📊 Scikit-learn
- 📈 Matplotlib
- 🎨 Seaborn

---

## 🎧 Feature Extraction

MFCC (**Mel-Frequency Cepstral Coefficients**) features were extracted from the speech audio files.

### 📊 Feature Details

- 🔹 **Feature:** MFCC
- 🔹 **MFCC Features:** 40
- 🔹 **Time Frames:** 200
- 🔹 **CNN Input Shape:** `(40, 200, 1)`

The extracted MFCC features were used as input to the CNN model for emotion classification.

---

## 🔄 Data Augmentation

Audio data augmentation was applied to increase the amount of training data and improve model generalization.

### 📊 Dataset Details

| 📌 Dataset | 🔢 Samples |
|---|---:|
| 🎧 Original Samples | 1440 |
| 🔄 Augmented Samples | 2880 |
| 🏋️ Original Training Samples | 1152 |
| 🚀 Final Training Samples | 2304 |
| 🧪 Testing Samples | 288 |

Audio augmentation helped the model learn more variations in speech and significantly improved the final performance.

---

## 🧠 Model

A **Convolutional Neural Network (CNN)** was used for speech emotion classification.

The project includes two models:

### 🔹 Baseline CNN

The baseline CNN was trained using the original training dataset.

### 🚀 Augmented CNN

The augmented CNN was trained using the augmented training dataset.

The augmented model performed significantly better than the baseline model.

---

## 📐 CNN Input Shape

```text
(40, 200, 1)
