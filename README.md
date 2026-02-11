# Bessel Encoded Quantum-Classic Hybrid Model (BEQCHM)
Code of the paper title "Bessel-Encoded Hybrid Quantum–Classical Architecture for Oscillatory Neural Time-Series Disease Classification" is present in this repository.

# Content
BEQCHM-1.py: Python file containing a model used to train BEQCHM for condition 1. In this condition one Bessel Encoded Variational Quantum Circuit (BE-VQC) is utilized.

BEQCHM-2.py: Python file containing a model used to train BEQCHM for condition 2. Two BE-VQCs are utilized in this condition.

BEQCHM-3.py: Python file containing a model used to train BEQCHM for condition 3. Three BE-VQCs are utilized in condition 3.

Addition_Baseline(FCLs replaces BE-VQCs)\: Python files implementing baseline Model for condition 1, 2, and 3 in which BE-VQC is replaced with single, double and three Fully connected layers

requirements.txt: Contains all the packages required to run this code 

# Dataset 
The current analysis included the initial rs-fMRI scans of 110 participants (54 healthy controls and 56 EMCI patients) in each of the multiple internal cohorts of ADNI (ADNI1, ADNI2, ADNIGO, and ADNI3). The resting-state fMRI images were selected specifically against three constant protocols: 140 time points with a repetition time (TR) of 3000 milliseconds and 48 anatomical slices. Based on this, the present investigation included resting-state fMRI data of the ADNI2, ADNIGO, and ADNI3 sub-cohorts. 


# Installation and Setup

For the full hybrid quantum-classical project, including all core, PyTorch, and quantum packages, you can install everything at once using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

# Authors 
Muhammad Saad, Wenjie Liu, Qingshan Wu
