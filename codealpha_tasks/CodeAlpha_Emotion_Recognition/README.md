# 🎙️ Speech Emotion Recognition using MFCC and CNN

## 📌 Project Overview

This project focuses on **Speech Emotion Recognition (SER)** using **Machine Learning and Deep Learning** techniques.

The system analyzes speech audio signals and predicts the emotional state of the speaker using **MFCC (Mel-Frequency Cepstral Coefficients)** as audio features and a **Convolutional Neural Network (CNN)** for classification.

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset and classifies speech into **8 different emotion categories**.

### 🎯 Objective

The main objective of this project is to:

* 🎙️ Process human speech audio signals
* 🔊 Extract meaningful speech features using MFCC
* 🧹 Convert variable-length audio features into a fixed-size representation
* 🔄 Improve model performance using audio data augmentation
* 🧠 Build a CNN-based emotion classification model
* 📊 Evaluate model performance using multiple classification metrics
* 🔍 Predict emotions from individual audio files
* 📈 Compare the performance of a baseline CNN and an augmented CNN

---

## 🗂️ Dataset

### RAVDESS Dataset

The project uses the **RAVDESS speech dataset** containing:

* 🎵 **1440 original audio files**
* 👥 **24 actors**
* 🎭 **8 emotion classes**

### Emotion Classes

| Label | Emotion   |
| ----: | --------- |
|     1 | Neutral   |
|     2 | Calm      |
|     3 | Happy     |
|     4 | Sad       |
|     5 | Angry     |
|     6 | Fearful   |
|     7 | Disgust   |
|     8 | Surprised |

---

## 🔄 Project Workflow

The complete pipeline used in this project is:

```text
Raw Speech Audio
       ↓
RAVDESS Dataset
       ↓
Audio Loading using Librosa
       ↓
MFCC Feature Extraction
       ↓
Fixed Size Feature Representation
       ↓
Train-Test Split
       ↓
Baseline CNN
       ↓
Audio Data Augmentation
       ↓
Augmented MFCC Features
       ↓
Final CNN Model
       ↓
Model Evaluation
       ↓
Emotion Prediction
```

---

## 🧹 Data Preprocessing

The audio files are loaded using **Librosa** with a sampling rate of:

```text
22050 Hz
```

For every audio file:

1. 🎵 Audio signal is loaded.
2. 🔊 MFCC features are extracted.
3. 📐 **40 MFCC coefficients** are generated.
4. ⏱️ The time dimension is standardized to **200 frames**.
5. ➕ If the MFCC contains fewer than 200 frames, zero-padding is applied.
6. ✂️ If the MFCC contains more than 200 frames, it is truncated to 200 frames.
7. 🧠 A channel dimension is added for CNN processing.

### Final Feature Shape

```text
MFCC Shape: (40, 200)
CNN Input Shape: (40, 200, 1)
```

---

## 🎵 Feature Extraction — MFCC

The project uses **Mel-Frequency Cepstral Coefficients (MFCC)** to represent important characteristics of the speech signal.

MFCCs are widely used for speech-related tasks because they capture important information about the spectral characteristics of human speech.

The project extracts:

```text
Number of MFCC Features = 40
Number of Time Frames   = 200
```

Therefore, every audio sample is represented as:

```text
(40, 200)
```

and converted into:

```text
(40, 200, 1)
```

for CNN input.

---

## 🔄 Data Augmentation

To improve the generalization ability of the model, audio augmentation was applied to the training data.

Three augmentation techniques were implemented:

### 1. 🔊 Noise Addition

Small Gaussian noise is added to the original audio signal.

```text
audio_aug = audio + 0.005 × noise
```

### 2. ⏩ Time Shifting

The audio signal is randomly shifted in time within approximately ±20% of the sampling rate.

### 3. 🔉 Volume Variation

The amplitude of the audio signal is randomly changed using a factor between:

```text
0.8 and 1.2
```

After augmentation, the signal is clipped to the valid range:

```text
[-1.0, 1.0]
```

### Augmented Dataset

```text
Original Samples   : 1440
Original + Augmented Samples : 2880
```

Importantly, the **test set remains untouched original data**, while augmented versions corresponding to training samples are used for training.

---

## 🧠 CNN Architecture

The final model is a **Convolutional Neural Network (CNN)** designed to classify the 8 emotion categories.

### Architecture

```text
Input
(40, 200, 1)
      ↓
Conv2D
32 Filters
3 × 3 Kernel
ReLU
      ↓
Batch Normalization
      ↓
MaxPooling2D
2 × 2
      ↓
Conv2D
64 Filters
3 × 3 Kernel
ReLU
      ↓
Batch Normalization
      ↓
MaxPooling2D
2 × 2
      ↓
Conv2D
128 Filters
3 × 3 Kernel
ReLU
      ↓
Batch Normalization
      ↓
MaxPooling2D
2 × 2
      ↓
Flatten
      ↓
Dense
128 Neurons
ReLU
      ↓
Dropout
0.4
      ↓
Dense
8 Neurons
Softmax
      ↓
Emotion Prediction
```

### Model Components

| Layer               | Configuration          |
| ------------------- | ---------------------- |
| Input               | `(40, 200, 1)`         |
| Conv2D              | 32 filters, 3×3, ReLU  |
| Batch Normalization | Applied                |
| MaxPooling2D        | 2×2                    |
| Conv2D              | 64 filters, 3×3, ReLU  |
| Batch Normalization | Applied                |
| MaxPooling2D        | 2×2                    |
| Conv2D              | 128 filters, 3×3, ReLU |
| Batch Normalization | Applied                |
| MaxPooling2D        | 2×2                    |
| Flatten             | Feature vector         |
| Dense               | 128 neurons, ReLU      |
| Dropout             | 0.4                    |
| Output              | 8 neurons, Softmax     |

### Model Parameters

The CNN contains approximately:

```text
Total Parameters      : 1,225,224
Trainable Parameters  : 1,224,776
Non-Trainable         : 448
```

---

## ⚙️ Model Training

The final augmented CNN was trained using:

* 🧠 **Optimizer:** Adam
* 📉 **Learning Rate:** `0.0005`
* 📊 **Loss Function:** Sparse Categorical Crossentropy
* 🎯 **Metric:** Accuracy
* 🔢 **Batch Size:** 32
* 🔄 **Maximum Epochs:** 40
* 🛑 **Early Stopping:** Patience = 7
* 📉 **ReduceLROnPlateau:** Factor = 0.5
* 🔽 **Minimum Learning Rate:** `0.00001`
* 🧪 **Validation Split:** 20%

### Training Strategy

Two important callbacks were used:

#### EarlyStopping

Training stops when validation loss does not improve for 7 consecutive epochs and the best model weights are restored.

#### ReduceLROnPlateau

The learning rate is reduced by a factor of 0.5 when validation loss stops improving.

---

## 📊 Train-Test Split

For the original dataset:

```text
Total Samples : 1440
Training Data : 1152
Testing Data  : 288
```

The split was performed using:

```text
Test Size   : 20%
Random State: 42
Stratified Split: Yes
```

The final test set contains only **original, untouched audio samples**.

---

## 🆚 Model Comparison

Two CNN approaches were evaluated.

| Model         | Test Accuracy |  Test Loss |
| ------------- | ------------: | ---------: |
| Baseline CNN  |        52.78% |     1.4498 |
| Augmented CNN |    **86.11%** | **0.4779** |

### 📈 Improvement

The augmented CNN improved test accuracy by:

```text
86.11% - 52.78% = 33.33 percentage points
```

This demonstrates a significant improvement after applying audio data augmentation and training the final CNN model.

---

## 🏆 Final Model Performance

The final augmented CNN achieved:

| Metric               |      Score |
| -------------------- | ---------: |
| 🎯 Test Accuracy     | **86.11%** |
| 📉 Test Loss         | **0.4779** |
| 📊 Macro F1-Score    | **85.71%** |
| 📊 Weighted F1-Score | **86.19%** |

### Classification Report

| Emotion   | Precision | Recall | F1-Score |
| --------- | --------: | -----: | -------: |
| Neutral   |    70.83% | 89.47% |   79.07% |
| Calm      |    89.47% | 89.47% |   89.47% |
| Happy     |    93.94% | 81.58% |   87.32% |
| Sad       |    76.32% | 76.32% |   76.32% |
| Angry     |    89.19% | 84.62% |   86.84% |
| Fearful   |    90.00% | 92.31% |   91.14% |
| Disgust   |    82.93% | 89.47% |   86.08% |
| Surprised |    91.89% | 87.18% |   89.47% |

---

## 🔍 Additional Audio Testing

After training, the final model was also evaluated across all **1440 RAVDESS audio files**.

```text
Total Audio Files Tested : 1440
Correct Predictions      : 1333
Incorrect Predictions    : 107
Overall Accuracy         : 92.57%
```

### Emotion-Wise Accuracy

| Emotion      | Accuracy |
| ------------ | -------: |
| 😠 Angry     |   95.31% |
| 😌 Calm      |   96.35% |
| 🤢 Disgust   |   92.71% |
| 😨 Fearful   |   88.02% |
| 😊 Happy     |   90.62% |
| 😐 Neutral   |   90.62% |
| 😢 Sad       |   89.58% |
| 😮 Surprised |   96.35% |

> **Note:** The official held-out test-set accuracy is **86.11%**. The additional 1440-file evaluation is reported separately because it is a different evaluation procedure.

---

## 🔮 Emotion Prediction Pipeline

The trained model can predict emotion from an individual `.wav` file.

The prediction pipeline is:

```text
Audio File
    ↓
Load Audio @ 22050 Hz
    ↓
Extract 40 MFCC Features
    ↓
Pad / Truncate to 200 Frames
    ↓
Add Batch + Channel Dimensions
    ↓
CNN Model
    ↓
Softmax Probabilities
    ↓
Highest Probability Class
    ↓
Predicted Emotion + Confidence
```

### Example Output

```text
========================================
         SPEECH EMOTION RESULT
========================================
Audio File : example.wav
Predicted Emotion : happy
Confidence        : XX.XX%
========================================
```

---

## 💾 Saved Model

The final trained model is saved as:

```text
final_emotion_cnn_86_11.keras
```

The project also stores emotion labels in:

```text
emotion_labels.json
```

The final model can be loaded using TensorFlow/Keras.

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "final_emotion_cnn_86_11.keras"
)
```

---

## 🛠️ Technologies Used

### Programming Language

* 🐍 Python

### Libraries & Frameworks

* TensorFlow / Keras
* Librosa
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* JSON

### Machine Learning / Deep Learning Concepts

* 🎙️ Speech Processing
* 🎵 MFCC Feature Extraction
* 🔄 Audio Data Augmentation
* 🧠 Convolutional Neural Networks
* 📊 Classification
* 📈 Model Evaluation
* 🔍 Confusion Matrix
* 📋 Classification Report

---

## 📁 Project Structure

A recommended GitHub project structure is:

```text
Emotion-Recognition/
│
├── 📓 Emotion_Recognition.ipynb
│
├── 🤖 final_emotion_cnn_86_11.keras
│
├── 🏷️ emotion_labels.json
│
├── 📊 results/
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── model_comparison.png
│
├── 📂 dataset/
│   └── RAVDESS/
│
└── 📄 README.md
```

> The RAVDESS audio dataset itself may be kept outside the GitHub repository if it is too large. The notebook can be used with the dataset path configured locally or through Google Drive.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Emotion-Recognition
```

### 2️⃣ Install Required Libraries

```bash
pip install tensorflow librosa numpy pandas scikit-learn matplotlib seaborn
```

### 3️⃣ Prepare the Dataset

Download and extract the **RAVDESS speech dataset**.

Place the dataset in the appropriate project directory.

### 4️⃣ Open the Notebook

Open:

```text
Emotion_Recognition.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 5️⃣ Run the Notebook

Execute the notebook cells in sequence to:

```text
Load Dataset
      ↓
Preprocess Audio
      ↓
Extract MFCC
      ↓
Train Baseline CNN
      ↓
Apply Data Augmentation
      ↓
Train Final CNN
      ↓
Evaluate Model
      ↓
Predict Emotions
```

---

## 📌 Key Learnings

Through this project, the following concepts were implemented:

* 🎙️ Speech signal processing
* 🎵 MFCC-based audio feature extraction
* 🧹 Audio preprocessing
* 🔄 Audio data augmentation
* 🧠 CNN architecture design
* ⚙️ Hyperparameter configuration
* 🛑 Early stopping
* 📉 Learning-rate scheduling
* 📊 Classification metrics
* 🔥 Confusion matrix analysis
* 🔮 Real audio emotion prediction
* 🧪 Model comparison and performance analysis

---

## 📈 Key Result

The project demonstrates the impact of data augmentation on speech emotion recognition.

```text
Baseline CNN
52.78%
    ↓
Data Augmentation
    ↓
Final Augmented CNN
86.11%
```

### ⭐ Performance Improvement

```text
+33.33 Percentage Points
```

The final CNN achieved **86.11% accuracy on the official held-out test set**, with a **Macro F1-Score of 85.71%**.

---

## 🔮 Future Improvements

Possible future improvements include:

* 🎤 Real-time microphone-based emotion recognition
* 🧠 CNN-LSTM or CRNN architecture
* 🔊 Spectrogram-based deep learning
* 🎵 Combining MFCC with Mel-Spectrogram features
* 🤖 Transfer learning using pretrained audio models
* 📱 Deploying the model as a web or mobile application
* ⚡ Real-time emotion prediction from live speech
* 📊 Further hyperparameter optimization

---

## ⚠️ Limitations

* The model is trained specifically on the RAVDESS dataset.
* Performance may vary on real-world speech recorded in different environments.
* Background noise, microphones, accents, speaking styles, and recording conditions may affect predictions.
* The model predicts only the 8 emotion categories represented in the dataset.

---

## 👨‍💻 Project Information

**Project:** Speech Emotion Recognition using MFCC and CNN

**Dataset:** RAVDESS

**Feature Extraction:** MFCC

**Deep Learning Model:** CNN

**Final Test Accuracy:** **86.11%**

**Macro F1-Score:** **85.71%**

**Final Model:** `final_emotion_cnn_86_11.keras`

---

## ⭐ Conclusion

This project successfully implements a **Speech Emotion Recognition system using MFCC features and a CNN classifier**.

The baseline CNN achieved **52.78% test accuracy**, while the final CNN trained with audio data augmentation achieved **86.11% test accuracy**, representing an improvement of **33.33 percentage points**.

The project demonstrates how proper **audio preprocessing, feature extraction, data augmentation, CNN architecture design, and evaluation techniques** can significantly improve the performance of speech emotion classification.

---
