# Student Exam Score Prediction with Stacking Ensemble

## Project Overview
This project focuses on **predicting student exam scores** using a progressively enhanced machine learning approach:

**Linear Regression → LightGBM → Stacked Ensemble**

The solution is designed with a strong emphasis on:
- Leakage-safe preprocessing
- Robust feature engineering
- Proper out-of-fold (OOF) stacking
- Transparent evaluation
- Kaggle-ready deployment

The final stacked model achieves **improved generalization and Kaggle performance** over individual base models.

---

## Dataset Description
- **Source:** Kaggle (Student Exam Performance Regression Dataset)
- **Files Used:**
  - `train.csv` – Training data with target variable
  - `test.csv` – Test data for submission
  - `sample_submission.csv` – Submission format
- **Target Variable:** `exam_score` (continuous)

### Feature Categories
- **Demographic:** `age`, `gender`
- **Academic:** `study_hours`, `class_attendance`, `course`, `study_method`
- **Lifestyle:** `sleep_hours`, `sleep_quality`
- **Institutional:** `facility_rating`, `internet_access`
- **Assessment:** `exam_difficulty`

---

## Exploratory Data Analysis (EDA)
EDA was conducted to understand feature distributions and relationships:
- Histograms & KDE plots for numerical variables
- Boxplots for categorical variables vs exam score
- Regression plots for linear trend inspection
- Category frequency analysis to detect imbalance

---

## Preprocessing & Feature Engineering

### Encoding Strategy
- **Binary Encoding**
  - `internet_access` → {yes: 1, no: 0}
- **Target Mean Encoding (Ordinal Features)**
  - `exam_difficulty`
  - `facility_rating`
  - `sleep_quality`
- **One-Hot Encoding (Nominal Features)**
  - `study_method`, `course`, `gender`
  - `handle_unknown='ignore'`
- **Numerical Scaling**
  - StandardScaler applied to numerical features

### Engineered Interaction Features
Created **inside pipelines** to prevent leakage:
- **Quality-adjusted sleep hours**
  - `sleep_hours × sleep_quality_target_mean`
- **Attendance-weighted study hours**
  - `study_hours × class_attendance`

---

## Models & Progressive Learning Strategy

### 1️⃣ Linear Regression (Baseline)
- Captures global linear trends
- High interpretability
- Serves as a strong baseline and complementary learner

### 2️⃣ LightGBM Regressor
- Models non-linear interactions
- Handles threshold effects and complex feature relationships
- Tuned for stability and generalization

### 3️⃣ Stacked Ensemble (Final Model)
- **Base models:** Linear Regression + LightGBM
- **Meta-learner:** Ridge Regression
- Trained on **out-of-fold predictions** to avoid leakage
- Combines linear and non-linear learning strengths

---

## Evaluation Strategy

### Out-of-Fold (OOF) Stacking
- K-Fold CV used to generate unbiased base model predictions
- Pipelines cloned and refit per fold
- OOF prediction correlation checked to ensure complementarity

**OOF RMSE (Linear + LGBM Stack):** **8.778**

---

## Model Performance Comparison (Test Set)

| Model                             | RMSE      | MAE       | R²        |
|-----------------------------------|-----------|-----------|-----------|
| Linear Regression                 | 8.710     | 6.937     | 0.785     |
| LightGBM Regressor                | 8.575     | 6.814     | 0.792     |
| **Stacked Model (Linear + LGBM)** | **8.572** | **6.809** | **0.792** |

✅ The stacked model provides the **best overall performance**, with lower error and stable variance.

---

## Kaggle Performance
- **Public Leaderboard RMSE:** **8.746**
- Performance improvement achieved through:
  - Target mean encoding
  - Feature interactions
  - Leakage-safe stacking ensemble

---

## Diagnostics & Interpretability
- Actual vs Predicted plots (OOF)
- Residual vs Prediction analysis
- Residual distribution inspection
- Meta-model coefficients reveal contribution of base learners
  
---

## Tech Stack
- **Python**
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- lightgbm
- category_encoders
- joblib

---

## Model Persistence
- Model saved using **joblib**
  
---
 ## Author  

**Lavan Kumar Konda**  
- Student at NIT Andhra Pradesh  
- Passionate about Data Science, Machine Learning, and AI  
- [LinkedIn](https://www.linkedin.com/in/lavan-kumar-konda/)
