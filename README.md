# Student Performance Prediction  
### An End-to-End Machine Learning Pipeline with Stacking Ensemble

## Overview

Educational institutions often struggle to identify students who may underperform until final evaluations are conducted.  
This project builds an **end-to-end machine learning system** to predict student exam scores using demographic, academic, and behavioral features.

Beyond achieving competitive predictive performance, the focus of this project is on:
- Robust evaluation using cross-validation
- Interpretability and modeling rationale
- Reproducible and deployment-ready ML pipelines

The solution progresses from simple, interpretable models to a carefully designed ensemble to balance performance and stability.

---
## Dataset Information

This project uses the **Student Test Scores Prediction** dataset from Kaggle.
- **Dataset Size**: ~630,000 samples

A small held-out test set is used due to the large dataset size, ensuring sufficient data for training while maintaining reliable evaluation.

Due to Kaggle’s data usage and licensing policies, the raw dataset files are
**not included** in this repository.

### Reproducing the results

To run the notebook locally, users should:

1. Download the dataset from the Kaggle competition  
   **Student Test Scores Prediction**
2. Extract the files `train.csv` and `test.csv`
3. Place them in the following directory structure

---

## Approach

### 1. Exploratory Data Analysis (EDA)

EDA was performed to understand:
- Target distribution
- Feature distributions
- Relationships between key features and exam scores

Key observations from EDA guided:
- Feature selection
- Encoding strategies
- Model choice

---

### 2. Feature Engineering & Preprocessing

The preprocessing pipeline includes:

- **Ordinal categorical features**  
  → Target Mean Encoding  
  (`exam_difficulty`, `facility_rating`, `sleep_quality`)

- **Nominal categorical features**  
  → One-Hot Encoding  
  (`study_method`, `course`, `gender`)

- **Binary feature**  
  → Manual mapping  
  (`internet_access`)

- **Numerical features**  
  → Standard Scaling  

#### Custom Interaction Features
Domain-inspired features were introduced:
- **Quality-adjusted sleep hours**  
- **Attendance-weighted study hours**

These interactions help capture real-world effects that linear and tree-based models exploit differently.

---

## Modeling Strategy

The modeling followed a **progressive and explainable approach**.

### Baseline Model — Linear Regression
- Captures global linear trends
- High interpretability
- Strong baseline performance

### Non-Linear Model — LightGBM Regressor
- Models non-linear interactions and threshold effects
- Controlled via explicit regularization and depth constraints
- Improves performance on complex patterns

### Final Model — Stacked Ensemble
- Combines Linear Regression and LightGBM using a Ridge meta-model
- Uses **out-of-fold (OOF) predictions** to prevent leakage
- Improves stability and generalization rather than aggressively chasing leaderboard gains

---

## Evaluation Strategy

- **Primary metric**: RMSE (competition metric)
- Cross-validation with OOF predictions for stacking
- Separate held-out test set for final evaluation
- Residual diagnostics to assess bias and variance behavior

---

## Model Performance (Test Set)

| Model | RMSE | MAE | R² |
|------|------|------|----|
| Linear Regression | — | — | — |
| LightGBM Regressor | — | — | — |
| Stacked Model (Linear + LightGBM) | — | — | — |

---

## Key Learnings

- Simple linear models can perform strongly when features are well-designed
- Non-linear models add value by capturing interactions and threshold effects
- Stacking improves **stability and generalization**, not just raw performance
- Proper cross-validation and OOF predictions are critical for reliable ensembles

---

## Inference & Deployment Readiness

- Unified inference function ensures training–inference consistency
- Full model bundle (base pipelines + meta-model) is serialized using `joblib`
- Ready for:
  - Batch prediction
  - REST API deployment (e.g., FastAPI)
  - Integration into larger systems

---

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- LightGBM
- category-encoders
- Matplotlib, Seaborn

---

## Future Improvements

- REST API deployment using FastAPI
- Containerization using Docker

---

## Repository Structure

student-performance-prediction/
│
├── notebook/
│   └── student_score_prediction.ipynb
│
├── artifacts/
│   └── student_score_stacked_model.joblib
│
├── outputs/
│   ├── model_comparison_metrics.csv
│   └── kaggle_test_submission.csv
│
├── data/
│   └── README.md          # Instructions to download dataset
│
├── .gitignore
│
├── requirements.txt
│
└── README.md

## Author  

**Lavan Kumar Konda**  
- Student at NIT Andhra Pradesh  
- Passionate about Data Science, Machine Learning, and AI  
- [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
