# Emotion Recognition from Speech Using MFCC and CNN

## Project Overview

This project is a Speech Emotion Recognition system that uses Machine Learning and Deep Learning techniques to identify emotions from human speech.

The system extracts MFCC (Mel-Frequency Cepstral Coefficients) features from speech audio and uses a Convolutional Neural Network (CNN) to classify the speech into different emotion categories.

## Dataset

The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

- Total Audio Files: 1440
- Number of Emotion Classes: 8

### Emotion Classes

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Technologies Used

- Python
- Google Colab
- TensorFlow
- Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Feature Extraction

MFCC features were extracted from the audio files.

- MFCC Features: 40
- Time Frames: 200
- CNN Input Shape: (40, 200, 1)

## Data Augmentation

Audio data augmentation was applied to increase the amount of training data and improve model generalization.

- Original Samples: 1440
- Augmented Samples: 2880
- Original Training Samples: 1152
- Final Training Samples: 2304
- Testing Samples: 288

## Model

A Convolutional Neural Network (CNN) was used for emotion classification.

The baseline CNN was first trained and evaluated. After that, an augmented dataset was used to train the final CNN model.

## Model Performance

| Model | Test Accuracy | Test Loss |
|---|---:|---:|
| Baseline CNN | 52.78% | 1.4498 |
| Augmented CNN | 86.11% | 0.4779 |

Data augmentation improved the test accuracy from 52.78% to 86.11%, an improvement of 33.33 percentage points.

## Final Model Results

- Test Accuracy: 86.11%
- Macro F1-Score: 85.71%
- Weighted F1-Score: 86.19%
- Test Loss: 0.4779

## Emotion-wise Accuracy

| Emotion | Accuracy |
|---|---:|
| Angry | 95.31% |
| Calm | 96.35% |
| Disgust | 92.71% |
| Fearful | 88.02% |
| Happy | 90.62% |
| Neutral | 90.62% |
| Sad | 89.58% |
| Surprised | 96.35% |

## Project Workflow

Audio Dataset  
↓  
Audio Preprocessing  
↓  
MFCC Feature Extraction  
↓  
Data Augmentation  
↓  
CNN Model Training  
↓  
Model Evaluation  
↓  
Emotion Prediction

## Files Included

- `emotion_recognition.ipynb` - Complete project notebook
- `final_emotion_cnn_86_11.keras` - Trained CNN model
- `emotion_labels.json` - Emotion class labels

## Important Note

The RAVDESS dataset is not included in this repository due to its size and licensing considerations.

## Conclusion

The final augmented CNN model achieved 86.11% accuracy on the official test set. The results demonstrate that MFCC-based feature extraction combined with CNN and audio data augmentation can effectively recognize emotions from speech.

## Internship

This project was completed as part of the CodeAlpha Internship.

**Repository:** `codealpha_tasks`