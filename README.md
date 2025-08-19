Federated Learning with Differential Privacy – Breast Cancer Dataset
 Overview

This notebook demonstrates:

Centralized training on the Breast Cancer Wisconsin dataset.

Federated Learning (FL) with two simulated hospitals using Flower
.

Federated Learning + Differential Privacy (DP) using Opacus
.

The goal is to compare privacy–utility tradeoffs across setups.

 Setup Instructions
1. Create Environment
conda create -n fl_dp python=3.10 -y
conda activate fl_dp

2. Install Dependencies
pip install torch torchvision torchaudio
pip install scikit-learn
pip install matplotlib seaborn
pip install flwr
pip install opacus


(If using Jupyter:)

pip install jupyter

 Running the Notebook

Open the environment:

conda activate fl_dp
jupyter notebook


Launch the provided notebook assignment4_federated_privacy.ipynb.

Run all cells from top to bottom.

 Expected Outputs

Centralized baseline model

Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

Federated model (no-DP)

Similar metrics, slight drop compared to centralized

Federated model (with DP)

Metrics with larger drop due to noise

Privacy budget ε and DP parameters (C, σ)

Comparative results saved to results\comparative_results.csv
                          Model  Accuracy  Precision  Recall      F1  ROC-AUC  \
Method                                                                          
Centralized  LogisticRegression    0.9737     1.0000  0.9286  0.9630   0.9964   
Fed (no-DP)            SmallMLP    0.3684     0.3684  1.0000  0.5385   0.9765   
Fed (DP)               SmallMLP    0.6316     0.0000  0.0000  0.0000   0.0182   

Plots:

Test ROC-AUC vs training rounds (Centralized vs FL vs FL+DP)

Artifacts:

models/fed_dp_final.pt (saved DP model)

 Reproducibility Checklist

Random seeds fixed (numpy, torch, sklearn)

Dependencies listed above (tested with Python 3.10)

Notebook runs top-to-bottom without manual intervention