# Domain-Adversarial Deep Learning for Cross-Dataset Cardiac Arrhythmia Detection

## Overview

This project focuses on automated cardiac arrhythmia detection using deep learning on multi-lead electrocardiogram (ECG) signals. Two convolutional neural network (CNN) architectures were developed and evaluated using combined ECG data from the PTB-XL and Lobachevsky University Electrocardiography Database (LUDB).

The goal was to improve cross-dataset generalization by training on heterogeneous clinical ECG sources rather than relying on a single dataset. A baseline 1D CNN was first implemented, followed by a stronger residual CNN with deeper feature extraction, residual connections, and improved regularization.

The improved model achieved significantly better classification performance and demonstrated stronger robustness across diagnostic classes.

---

## Problem Statement

Accurate arrhythmia detection from ECG signals is critical for early diagnosis and prevention of serious cardiac events. While deep learning models have shown strong performance in ECG classification, many models fail to generalize well across datasets due to differences in:

- acquisition hardware
- annotation standards
- patient demographics
- signal quality
- clinical environments

This project addresses that limitation by combining multiple ECG datasets and evaluating whether deeper CNN architectures improve robustness and classification accuracy across heterogeneous data sources.

---

## Dataset

### PTB-XL

The PTB-XL dataset is a large-scale 12-lead clinical ECG dataset containing:

- 21,837 ECG recordings
- 18,885 patients
- 10-second recordings
- 500 Hz sampling frequency
- SCP-ECG diagnostic labels across 71 classes

Each ECG contains physician-verified annotations and diagnostic metadata.

### LUDB (Lobachevsky University Database)

The LUDB dataset provides:

- multi-lead ECG recordings
- additional rhythm diversity
- different acquisition conditions
- varied patient populations
- external domain variation for robustness testing

This dataset helps simulate real-world deployment where incoming ECGs may differ significantly from the training source.

---

## Scope

This project includes:

- loading and preprocessing ECG waveform data from PTB-XL and LUDB
- signal standardization and alignment
- label extraction and encoding
- multi-dataset merging for cross-domain learning
- baseline 1D CNN development
- improved residual CNN development
- model training and validation
- evaluation using:
  - test accuracy
  - confusion matrices
  - ROC curves
  - AUC scores
  - precision-recall curves

This project does not include:

- transformer-based ECG models
- domain adversarial loss functions
- real-time deployment systems
- clinician-facing inference pipelines

---

## Key Variables

### Input Features

- 12 ECG leads
- 5000 timesteps per signal (trimmed to 3000 for improved model)
- float16 memory optimization
- shape converted to:

```python
(samples, timesteps, channels)
```

Example:

```python
(samples, 3000, 12)
```

### Target Variable

Multi-class arrhythmia classification across:

- PTB-XL SCP diagnostic labels
- LUDB rhythm labels

Final cleaned label space:

- 52 diagnostic classes

Rare classes with fewer than 2 occurrences were removed.

---

## Tech Stack

### Languages

- Python

### Libraries

- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- WFDB

### Environment

- Anaconda
- Jupyter Notebook

### Model Types

- Baseline 1D CNN
- Residual 1D CNN (ResNet-inspired)

---

## Methodology

## 1. Data Loading

ECG records were loaded using the `wfdb` package from both PTB-XL and LUDB datasets.

Signals were extracted as:

```python
(12, 5000)
```

and converted to `float16` for memory efficiency.

---

## 2. Label Processing

PTB-XL labels were extracted from SCP diagnostic codes.

Example:

```python
{"NORM": 100}
→ PTB-NORM
```

LUDB labels were taken from rhythm annotations.

Labels were encoded using `LabelEncoder`.

Rare labels with fewer than 2 samples were removed.

---

## 3. Dataset Combination

Both datasets were aligned by:

- trimming signals to equal length
- matching lead ordering
- concatenating features and labels

Final cleaned dataset:

- 21,846 ECG samples
- 52 diagnostic classes

---

## 4. Baseline Model

A standard 1D CNN was developed using:

- Conv1D layers
- BatchNormalization
- MaxPooling1D
- Dropout
- Dense classification layers

This served as the benchmark model.

---

## 5. Improved Model

A stronger residual CNN was developed using:

- deeper convolutional stacks
- residual skip connections
- strided convolutions
- GlobalAveragePooling1D
- stronger dropout regularization

This architecture improved gradient flow and captured longer-range ECG dependencies.

---

## 6. Evaluation

Models were evaluated using:

- test accuracy
- confusion matrix
- ROC curves
- AUC
- precision-recall curves

This provided both threshold-dependent and threshold-independent performance analysis.

---

## Results

| Model | Test Accuracy |
|---|---:|
| Baseline 1D CNN | 54.62% |
| Residual CNN | 66.25% |

### Key Findings

- deeper architecture improved generalization
- residual connections reduced misclassification
- stronger feature extraction improved class separation
- multi-dataset training improved robustness across domains

The improved residual CNN demonstrated substantially better performance across nearly all evaluation metrics.

---

## How to Run

```bash
# Clone the repository
git clone <your-repo-link>

# Navigate into project folder
cd ECG-Arrhythmia-Detection

# Create environment (optional)
conda create -n ecg2 python=3.10
conda activate ecg2

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Required Dataset Setup

Download and place the following datasets locally:

- PTB-XL
- LUDB

Update the dataset paths in the notebook:

```python
ptb_root = "your_path/PTB-XL"
ludb_signal_path = "your_path/ludb/ludb"
ludb_csv_path = "your_path/ludb/ludb.csv"
```

---

## Project Structure

```text
CardiacArrhythmia_DeepLearning/
│
├── DomainAdversarial_DL_CrossDataset_Cardiac_Arrhythmia_Detection.ipynb
├── Keras_Model.keras
├── README.md
└── DomainAdversarial_DL_CrossDataset_Cardiac_Arrhythmia_Detection_Report.pdf
```

---

## Author

### Aaron Tsui

Master of Science in Data Science  
Northwestern University 

Specializing in machine learning, deep learning, and applied AI systems for healthcare and large-scale predictive modeling.

- Email: aaron.tsui.careers@gmail.com  
- LinkedIn: https://www.linkedin.com/in/aaron-tsui/
