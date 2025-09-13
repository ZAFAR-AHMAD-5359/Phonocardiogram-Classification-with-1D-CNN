# Phonocardiogram Classification with 1D-CNN and Pitch-Shifting Augmentation 🫀🔊

[![Paper](https://img.shields.io/badge/Paper-EUSIPCO%202024-blue)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)

## 📋 Overview

This repository implements a novel **1D Convolutional Neural Network** for automated classification of heart sounds (phonocardiograms) with an innovative **pitch-shifting augmentation** technique that preserves diagnostic frequencies while improving model robustness.

### 🎯 Key Features
- ✅ **99.6% Classification Accuracy** on test dataset
- ✅ **Novel Pitch-Shifting Augmentation** preserving 20-650 Hz diagnostic band
- ✅ **Class-Balanced Focal Loss** for handling imbalanced medical data
- ✅ **Real-time Processing** capability for clinical deployment
- ✅ **Multi-class Classification** for 8 cardiac conditions

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/ZAFAR-AHMAD-5359/Phonocardiogram-Classification-with-1D-CNN.git
cd Phonocardiogram-Classification-with-1D-CNN
```

### Basic Usage
```python
from pcg_classifier import PCGClassifier
import numpy as np

# Initialize classifier
classifier = PCGClassifier(model_path='models/best_model.h5')

# Load and preprocess PCG signal
signal = np.load('sample_pcg.npy')
preprocessed = classifier.preprocess(signal, sample_rate=4000)

# Classify
prediction, confidence = classifier.classify(preprocessed)
print(f"Diagnosis: {prediction} (Confidence: {confidence:.2%})")
```

## 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy** | 99.6% | Overall classification accuracy |
| **Sensitivity** | 98.2% | True positive rate for abnormal cases |
| **Specificity** | 99.8% | True negative rate for normal cases |
| **F1-Score** | 0.987 | Harmonic mean of precision and recall |
| **AUC-ROC** | 0.994 | Area under the ROC curve |

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

## 🔬 Methodology

### 1. Signal Preprocessing
```python
def preprocess_signal(signal, fs=4000):
    # Bandpass filter (20-650 Hz)
    filtered = butterworth_filter(signal, 20, 650, fs)

    # Normalize
    normalized = (filtered - np.mean(filtered)) / np.std(filtered)

    # Segment into 4-second chunks
    segments = segment_signal(normalized, chunk_size=4*fs)

    return segments
```

### 2. Pitch-Shifting Augmentation
Our novel approach preserves diagnostic frequencies:
```python
def pitch_shift_augmentation(signal, shift_range=[-2, 2]):
    # Preserve 20-650 Hz diagnostic band
    diagnostic_band = extract_band(signal, 20, 650)

    # Apply pitch shift to higher frequencies
    high_freq = extract_band(signal, 650, 2000)
    shifted = librosa.effects.pitch_shift(high_freq, sr=4000, n_steps=shift_range)

    # Combine preserved and shifted components
    augmented = diagnostic_band + shifted
    return augmented
```

### 3. 1D-CNN Architecture
```
Input (4000,1)
    ↓
Conv1D(64, 50) → BatchNorm → ReLU → MaxPool
    ↓
Conv1D(128, 25) → BatchNorm → ReLU → MaxPool
    ↓
Conv1D(256, 10) → BatchNorm → ReLU → GlobalMaxPool
    ↓
Dense(128) → Dropout(0.5)
    ↓
Dense(8, softmax)
```

## 📁 Dataset

### Data Distribution
| Class | Samples | Percentage |
|-------|---------|------------|
| Normal | 150 | 25.0% |
| VSD | 72 | 12.0% |
| ASD | 48 | 8.0% |
| MR | 36 | 6.0% |
| PDA | 24 | 4.0% |
| AS | 30 | 5.0% |
| PS | 27 | 4.5% |
| Others | 219 | 36.5% |
| **Total** | **606** | **100%** |

### Data Sources
- 🏥 Multi-site collection from 3 hospitals
- 👶 Pediatric population (ages 0-18)
- ✅ Echo-validated ground truth

## 🛠️ Advanced Features

### Feature Extraction
- **MFCC** (13 coefficients)
- **Spectral Centroid**
- **Zero Crossing Rate**
- **Spectral Roll-off**
- **Envelope Features**

### Model Optimization
- **Early Stopping** with patience=10
- **Learning Rate Scheduling**
- **K-Fold Cross Validation** (k=5)
- **Class Weights** for imbalanced data

## 📈 Results Visualization

### Training History
![Training History](assets/training_history.png)

### ROC Curves
![ROC Curves](assets/roc_curves.png)

### Sample Classifications
![Sample Results](assets/sample_results.png)

## 🔧 Configuration

Edit `config.yaml` for custom settings:
```yaml
model:
  architecture: "1D-CNN"
  input_shape: [4000, 1]
  num_classes: 8

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001

augmentation:
  pitch_shift: true
  shift_range: [-2, 2]
  preserve_band: [20, 650]
```

## 📚 Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{ahmad2024phonocardiogram,
  title={Phonocardiogram Classification Based on 1D CNN with Pitch-Shifting and Signal Uniformity Techniques},
  author={Ahmad, Zafar and Khan, Salman and others},
  booktitle={European Signal Processing Conference (EUSIPCO)},
  year={2024},
  organization={IEEE}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Zafar Ahmad**
- 🌐 GitHub: [@ZAFAR-AHMAD-5359](https://github.com/ZAFAR-AHMAD-5359)
- 📧 Email: zafarahmad5359@gmail.com
- 🎓 Google Scholar: [Publications](https://scholar.google.com/citations?user=D5W9TVwAAAAJ&hl=en)

## 🙏 Acknowledgments

- Qatar University for research support
- Medical staff at participating hospitals
- Open-source community for tools and libraries

## ⚠️ Disclaimer

This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis.

---
⭐ **If you find this work useful, please consider giving it a star!**