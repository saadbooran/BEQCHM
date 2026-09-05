# Bessel Encoded Quantum-Classic Hybrid Model (BEQCHM)
Code of the paper title "BEQCHM: An efficient hybrid Bessel-encoded multi-topology VQC-GRU fusion model for neural oscillatory recognition and early MCI classification" is present in this repository.

# Content
BEQCHM-1.py: Python file containing a model used to train BEQCHM for condition 1. In this condition one Bessel Encoded Variational Quantum Circuit (BE-VQC) is utilized.

BEQCHM-2.py: Python file containing a model used to train BEQCHM for condition 2. Two BE-VQCs are utilized in this condition.

BEQCHM-3.py: Python file containing a model used to train BEQCHM for condition 3. Three BE-VQCs are utilized in condition 3.

**Baseline.FCLs_replaces_BE_VQCs:** Python files implementing baseline models for conditions 1, 2, and 3, in which the BE-VQC is replaced with single, double, and triple fully connected layers.

**Ablations:** Python files implementing ablation variants of the BE-VQC encoding and entanglement topology.

**Classical_Baselines:** Python files implementing classical (non-quantum) baseline models for comparison.


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
