# 🗣️ SpeakSense – AI Language Detection Tool

## Overview
**SpeakSense** is an AI-powered language detection system that identifies spoken languages from audio recordings using machine learning and deep learning techniques. Designed as a final project for the Machine Learning course (CSCI 6364) at The George Washington University, this tool demonstrates the potential of AI in facilitating multilingual communication by detecting languages from diverse audio inputs across various accents, dialects, and environments.

## 🚀 Project Highlights
- Detects 13 languages from audio: English, Spanish, German, and 10 Indian languages (Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Urdu).
- Utilizes real-world and Kaggle-sourced datasets totaling over **120,000 audio samples**.
- Explores multiple approaches including:
  - Classical ML (Random Forest, Gradient Boosting, SVM)
  - Deep Neural Networks (DNNs)
  - Convolutional Neural Networks (CNNs)
  - Spectrogram-based CNNs
  - Ensemble models

## 📊 Methodology

### 🔍 Exploratory Data Analysis (EDA)
- Analyzed waveform and spectrograms to inspect labeling and quality issues.
- Resolved label inconsistencies (e.g., mislabeling in Punjabi dataset).
- Standardized all audio files to **10-second segments** at **22,050 Hz**.
- Employed techniques like **MFCC**, **Zero-Crossing Rate**, **Spectral Entropy**, **Pitch Analysis**, and more.

### 🛠️ Feature Engineering
- **Feature Matrix** (58 features × 431 time frames)
- **Feature Vector** (58 aggregated features per audio file)
- Applied `StandardScaler` for feature normalization.
- Attempted denoising filters (with limited improvement on validation accuracy).

### 🧠 Model Training
Five major training paradigms were used:
1. **Classical ML on Feature Vectors**
   Algorithms like Random Forest and SVM showed overfitting, with high train but poor test accuracy.
2. **DNN on Feature Vectors**
   Achieved robust performance and generalization (Test Acc: 99.3%, ROC AUC: 0.996 on scaled data).
3. **CNN on Feature Matrix**
   Strong accuracy with unscaled input; slightly weaker performance on recorded real-world samples.
4. **CNN on Spectrogram Images**
   Underperformed significantly with poor convergence.
5. **Ensemble Models**
   - *Approach 1*: Combined top-three class probabilities
   - *Approach 2*: Used Hadamard product of model outputs

## ⚙️ Performance Evaluation

### 📈 On Controlled Test Data
| Model | Accuracy | ROC AUC |
|-------|----------|---------|
| DNN (Scaled) | **99.3%** | **0.996** |
| CNN (Feature Matrix, Scaled) | 89.0% | 0.940 |
| CNN (Spectrogram) | 6.0% | 0.498 |

### 🎙️ On Real-World Audio (Recorded Samples)
| Model | Accuracy |
|-------|----------|
| DNN (Unscaled) | 61.9% |
| CNN (Unscaled) | 61.9% |
| DNN (Scaled) | 38.1% |
| CNN (Scaled) | 47.6% |
| Ensemble (Approach 1 & 2) | 61.9% |

> Real-world samples introduced domain shift (noise, accents, mic quality), limiting generalization.

## ⚠️ Challenges & Future Work
- **Generalization gap** between training data and real-world audio.
- Spectrogram CNN models failed to converge effectively.
- Potential improvements include:
  - More diverse, noisy, and realistic training data
  - Domain adaptation techniques
  - Fine-tuned ensemble models
  - End-to-end models with robust audio augmentation

## 🤖 Tech Stack
- Python, Jupyter Notebooks
- Scikit-learn, TensorFlow, Keras
- Librosa, NumPy, Pandas, Matplotlib, Seaborn

## 👨‍🏫 Team & Acknowledgments
Developed by:
- Abde Manaaf Ghadiali (G29583342)
- Gehna Ahuja (G35741419)
- Venkatesh Shanmugam (G27887303)

Under the guidance of **Prof. Sardar Hamidian** and **Prof. Armin Mehrabian**

## 📚 References
- Rabiner & Juang (1992). Hidden Markov Models for Speech Recognition.
- Juang & Rabiner (1990). Language identification in speech signals.
- Waibel & Lee (1990). Multilingual speech recognition.
- Bahdanau et al. (2016). Language recognition using deep learning.
- Cavnar & Trenkle (1994). Language identification from short texts.

## 🔗 Project Link
[🔗 SpeakSense – GitHub Repository](https://github.com/your-repo-link)
