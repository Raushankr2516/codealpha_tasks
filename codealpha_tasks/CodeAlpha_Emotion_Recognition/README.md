# 🎙️ Emotion Recognition from Speech Using MFCC and CNN

## 📌 Project Overview

This project is a **Speech Emotion Recognition system** that uses Machine Learning and Deep Learning techniques to identify emotions from human speech.

The system extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features from speech audio and uses a **Convolutional Neural Network (CNN)** to classify the speech into different emotion categories.

---

## 📂 Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

- 🎵 **Total Audio Files:** 1440
- 😊 **Number of Emotion Classes:** 8

### 🎭 Emotion Classes

- 😐 Neutral
- 😌 Calm
- 😄 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fearful
- 🤢 Disgust
- 😲 Surprised

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

MFCC features were extracted from the audio files.

- 🔹 **MFCC Features:** 40
- 🔹 **Time Frames:** 200
- 🔹 **CNN Input Shape:** `(40, 200, 1)`

MFCC features were used to represent important characteristics of the speech audio and provide suitable input to the CNN model.

---

## 🔄 Data Augmentation

Audio data augmentation was applied to increase the amount of training data and improve model generalization.

- 📌 **Original Samples:** 1440
- 🔄 **Augmented Samples:** 2880
- 🏋️ **Original Training Samples:** 1152
- 🚀 **Final Training Samples:** 2304
- 🧪 **Testing Samples:** 288

---

## 🧠 Model

A **Convolutional Neural Network (CNN)** was used for emotion classification.

The **Baseline CNN** was first trained and evaluated. After that, an **Augmented CNN** was trained using the augmented dataset to improve model performance.

### 🔹 CNN Input

```text
(40, 200, 1)
