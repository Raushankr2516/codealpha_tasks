# 🎙️ Emotion Recognition from Speech Using MFCC and CNN

## 📌 Project Overview

This project is a **Speech Emotion Recognition (SER)** system that uses Machine Learning and Deep Learning techniques to identify emotions from human speech.

The system extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features from speech audio and uses a **Convolutional Neural Network (CNN)** to classify the speech into different emotion categories.

---

## 🎯 Objectives

- 🎙️ Analyze human speech audio
- 🔊 Extract meaningful speech features using MFCC
- 🧠 Build a CNN-based emotion classification model
- 🔄 Improve model performance using audio data augmentation
- 📊 Evaluate the model using accuracy, precision, recall, and F1-score
- 🎯 Predict emotions from new audio files

---

## 📂 Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

- 🎵 **Total Audio Files:** 1440
- 😊 **Number of Emotion Classes:** 8

### 😊 Emotion Classes

| 🔢 | Emotion |
|---:|---|
| 1️⃣ | Neutral |
| 2️⃣ | Calm |
| 3️⃣ | Happy |
| 4️⃣ | Sad |
| 5️⃣ | Angry |
| 6️⃣ | Fearful |
| 7️⃣ | Disgust |
| 8️⃣ | Surprised |

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

**MFCC (Mel-Frequency Cepstral Coefficients)** features were extracted from the audio files.

- 🔹 **MFCC Features:** 40
- 🔹 **Time Frames:** 200
- 🔹 **CNN Input Shape:** `(40, 200, 1)`

MFCCs help represent important frequency characteristics of human speech and are widely used in speech processing applications.

---

## 🔄 Data Augmentation

Audio data augmentation was applied to increase the amount of training data and improve model generalization.

| 📊 Dataset | 🔢 Samples |
|---|---:|
| Original Dataset | 1440 |
| Augmented Dataset | 2880 |
| Original Training Samples | 1152 |
| Final Training Samples | 2304 |
| Testing Samples | 288 |

---

## 🧠 Model

A **Convolutional Neural Network (CNN)** was used for emotion classification.

The project first trained a **Baseline CNN** and then trained an **Augmented CNN** using the augmented training dataset.

### 🔹 CNN Input

```text
(40, 200, 1)
