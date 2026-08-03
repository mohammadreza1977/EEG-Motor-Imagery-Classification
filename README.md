# EEG-Based Single-Hand Motor Imagery Classification

## A Dual-Branch Deep Learning Framework with Custom Spatial-Channel Attention

This repository contains the implementation and experimental results of a deep learning framework for **EEG-based classification of single-hand Motor Imagery (MI) tasks**.

The proposed framework combines two complementary EEG representations — **Power** and **Phase Locking Value (PLV)** — using a **dual-branch convolutional neural network with a custom spatial-channel attention mechanism**.

The project was developed as part of research in **Biomedical Engineering, EEG Signal Processing, Brain-Computer Interfaces (BCI), and Medical AI**.

---

## 1. Research Objective

Motor Imagery Brain-Computer Interfaces provide a non-invasive way of decoding imagined movements from EEG signals.

While distinguishing between left- and right-hand Motor Imagery is relatively well studied, distinguishing between **different imagined movements performed with the same hand** is considerably more challenging.

This project investigates a deep learning approach for distinguishing multiple single-hand Motor Imagery tasks using complementary representations of EEG activity.

The main objective is to determine whether combining:

* localized spectral activity
* functional connectivity
* spatial information
* channel-level information

can improve the representation and classification of Motor Imagery EEG signals.

---

## 2. Dataset

The study uses a public EEG dataset containing:

* **25 participants**
* **11 unilateral Motor Imagery tasks**
* Multiple recording sessions
* EEG recordings from multiple scalp electrodes

The experiments in this implementation focus on three Motor Imagery tasks:

* **Reaching**
* **Multigrasp**
* **Twist**

The original dataset is not included in this repository.

Dataset documentation is available in:

```text
Data Note/
```

---

## 3. EEG Processing Pipeline

The implemented processing pipeline consists of several stages.

```text
Raw EEG
   │
   ▼
Channel Selection
   │
   ▼
Common Average Reference (CAR)
   │
   ▼
Downsampling
   │
   ▼
60 Hz Notch Filtering
   │
   ▼
8–30 Hz Bandpass Filtering
   │
   ▼
4-second Motor Imagery Trial
   │
   ▼
2-second Sliding Windows
   │
   ├───────────────────┐
   ▼                   ▼
Power Features       PLV Features
   │                   │
   ▼                   ▼
CNN Branch           CNN Branch
   │                   │
   ▼                   ▼
CBAM Attention       CBAM Attention
   │                   │
   └─────────┬─────────┘
             ▼
      Feature Fusion
             │
             ▼
       Fully Connected
             │
             ▼
         Classification
```

---

## 4. EEG Preprocessing

The implemented preprocessing pipeline includes:

### Channel Selection

A subset of EEG channels associated with the sensorimotor regions is selected for analysis.

### Common Average Referencing

Common Average Referencing (CAR) is applied as a spatial preprocessing step.

### Downsampling

The EEG signals are downsampled to:

```text
250 Hz
```

### Notch Filtering

A notch filter is applied around:

```text
60 Hz
```

to suppress power-line interference.

### Bandpass Filtering

A bandpass filter is applied in the:

```text
8–30 Hz
```

frequency range.

### Temporal Windowing

The Motor Imagery trials are segmented into overlapping windows.

The implementation uses:

```text
Trial duration: 4 seconds
Window length: 2 seconds
```

---

# 5. EEG Feature Representations

Two complementary representations of EEG activity are used.

## 5.1 Power Representation

Power-based features represent the spectral activity of EEG signals and provide information about localized cortical activation.

The representation is particularly relevant to Motor Imagery because MI-related activity is associated with changes in sensorimotor rhythms.

---

## 5.2 Phase Locking Value (PLV)

Phase Locking Value is used to characterize the phase synchronization between EEG channels.

PLV therefore provides information about **functional connectivity** between different brain regions.

Instead of relying exclusively on local signal activity, the model can therefore learn relationships between EEG channels.

---

# 6. Proposed Deep Learning Architecture

The proposed model is a **Dual-Branch CNN architecture**.

Each branch processes a different EEG representation:

```text
                 ┌───────────────┐
                 │ Power Matrix  │
                 └───────┬───────┘
                         │
                         ▼
                    CNN Branch
                         │
                         ▼
                       CBAM
                         │
                         ▼
                      GAP + FC
                         │
                         │
                         ├──────────────┐
                         │              │
                         │              ▼
                         │        Feature Fusion
                         │              ▲
                         │              │
                         ▼              │
                    CNN Branch         │
                         ▲              │
                         │              │
                 ┌───────┴───────┐      │
                 │   PLV Matrix  │──────┘
                 └───────────────┘
```

The two branches learn complementary representations before their high-level features are fused.

---

# 7. Custom Spatial-Channel Attention

A major component of the proposed architecture is the **Convolutional Block Attention Module (CBAM)**.

CBAM contains two attention mechanisms.

## Channel Attention

Channel attention learns which feature channels contain the most informative representations.

This allows the network to emphasize relevant feature maps and suppress less useful information.

## Spatial Attention

Spatial attention learns which spatial regions of the feature representation are most informative.

This is particularly useful for EEG representations where discriminative spatial patterns can correspond to different sensorimotor activity distributions.

The combination of channel and spatial attention allows the model to adaptively focus on the most informative features.

---

# 8. Feature Fusion

After independent processing of Power and PLV representations, the two learned feature vectors are combined.

The implementation uses:

```text
Power branch → FC(64)
PLV branch   → FC(64)

              ↓
       Feature Concatenation
              ↓

            FC(128)
              ↓
             ReLU
              ↓
          Dropout(0.6)
              ↓
       Output Classification
```

This allows the classifier to exploit both:

* local/spectral EEG information
* functional connectivity information

within a unified representation.

---

# 9. Training Configuration

The model was implemented using:

```text
MATLAB R2022b
Deep Learning Toolbox
```

### Main Training Parameters

| Parameter                 | Configuration |
| ------------------------- | ------------- |
| Optimizer                 | Adam          |
| Initial Learning Rate     | 0.0005        |
| Maximum Epochs            | 100           |
| Batch Size                | 64            |
| Dropout                   | 0.6           |
| Learning Rate Drop Factor | 0.5           |
| Learning Rate Drop Period | 15            |
| Cross-Validation          | 10-Fold       |
| Early Stopping Patience   | 15            |

The implementation also uses validation monitoring and learning-rate scheduling during training.

---

# 10. Evaluation

The experimental framework uses **10-fold cross-validation**.

The implementation keeps overlapping windows originating from the same trial together during fold assignment to reduce the risk of information leakage between training and evaluation data.

Performance is evaluated using classification accuracy and confusion matrices.

---

# 11. Experimental Results

The proposed framework was evaluated under different classification settings.

| Classification Setting     | Mean Accuracy |
| -------------------------- | ------------: |
| Binary Classification      |    **72.18%** |
| Three-Class Classification |    **54.87%** |
| Six-Class Classification   |    **31.37%** |

The results demonstrate that classification becomes substantially more challenging as the number of Motor Imagery classes increases.

This is particularly relevant for single-hand MI because different imagined movements can produce highly similar EEG patterns.

---

# 12. Results Visualization

The repository contains the experimental results and visualization outputs.

### Confusion Matrices

```text
Results_Final_Optimized4444/
│
├── Overall_CM_Task_Grasp.png
├── Overall_CM_Task_Reach.png
└── Overall_CM_Task_Twist.png
```

### Performance Visualization

```text
├── Session_Accuracy_LinePlot.png
├── Task_Accuracy_BarPlot.png
└── Task_Accuracy_BoxPlot.png
```

These visualizations provide insight into task-wise and session-wise classification performance.

---

# 13. Repository Structure

```text
EEG-Motor-Imagery-Classification/
│
├── README.md
│
├── Data Note/
│   └── Dataset documentation
│
├── Results_Final_Optimized4444/
│   ├── FeatureCache/
│   ├── SavedNets/
│   ├── Overall_CM_Task_Grasp.png
│   ├── Overall_CM_Task_Reach.png
│   ├── Overall_CM_Task_Twist.png
│   ├── Session_Accuracy_LinePlot.png
│   ├── Task_Accuracy_BarPlot.png
│   ├── Task_Accuracy_BoxPlot.png
│   └── results tables
│
├── Structure/
│   ├── Slide1.JPG
│   └── Slide1.PNG
│
├── code/
│   └── Optimized4444.m
│
└── .gitignore
```

---

# 14. Technologies

### Programming

* MATLAB

### Deep Learning

* Convolutional Neural Networks
* CBAM Attention
* Feature Fusion
* Adam Optimization
* Early Stopping

### EEG / BCI

* EEG Signal Processing
* Motor Imagery
* Spectral Power Representation
* Phase Locking Value (PLV)
* Functional Connectivity
* Brain-Computer Interfaces

### Research Areas

* Biomedical Signal Processing
* Deep Learning
* Medical AI
* Neural Signal Processing
* Brain-Computer Interfaces

---

# 15. Research Contributions

The main research contributions of this project include:

1. Development of a **dual-branch deep learning architecture** for EEG classification.
2. Joint utilization of **Power and PLV representations**.
3. Integration of **spatial and channel attention** using CBAM.
4. Fusion of complementary EEG representations at the learned feature level.
5. Evaluation of the framework across different Motor Imagery classification settings.
6. Investigation of the increasing classification difficulty associated with a larger number of single-hand MI classes.

---

# 16. Limitations

The study highlights several challenges associated with EEG-based Motor Imagery classification:

* High inter-subject variability
* Similarity between neural patterns of different Motor Imagery tasks
* Reduced performance as the number of classes increases
* Limited generalization across participants
* Difficulty of robust classification in complex multi-class settings

---

# 17. Future Directions

Potential future research directions include:

* Improving cross-subject generalization
* Increasing the number and diversity of participants
* Exploring more advanced attention mechanisms
* Investigating stronger data augmentation strategies
* Developing real-time EEG classification systems
* Evaluating the model on larger and more diverse datasets
* Exploring efficient deployment for practical BCI applications

---

# 18. Research Application

This work contributes to the development of intelligent EEG-based Brain-Computer Interface systems.

Potential applications include:

* Neurorehabilitation
* Assistive technologies
* Motor intention decoding
* Human-computer interaction
* Brain-controlled systems
* Biomedical AI

---

# 19. Reproducibility

The repository provides the MATLAB implementation and selected experimental outputs.

The original EEG dataset is not included due to dataset distribution considerations.

To reproduce the experiments, the required dataset must first be obtained from its original source and configured according to the data-loading structure used in the MATLAB implementation.

---

# 20. Citation

If you use this work in academic research, please cite the associated thesis:

**Madani, Seyed Mohammad Reza.**

*Classification of Different Single-Hand Motor Imagery Tasks Using Electroencephalogram Signals Based on Deep Learning.*

M.Sc. Thesis, Biomedical Engineering – Bioelectric, Iran University of Science and Technology (IUST), Tehran, Iran, Tehran, Iran.

### Related Research Article

*A Dual-Branch Multimodal Deep Learning Framework with Custom Spatial-Channel Attention for EEG-Based Single-Hand Motor Imagery Classification.*

---

# Author

**Seyed Mohammad Reza Madani**

M.Sc. Biomedical Engineering – Bioelectric

**Iran University of Science and Technology (IUST), Tehran, Iran**

### Research Interests

* Artificial Intelligence in Medicine
* Deep Learning
* EEG Signal Processing
* Brain-Computer Interfaces
* Biomedical Signal Processing
* Medical AI
* Neural Signal Processing
