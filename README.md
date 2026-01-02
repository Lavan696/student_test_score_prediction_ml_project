# Student Exam Score Prediction (Regression)

## Project Overview
This project builds a **leakage-safe machine learning pipeline** to predict **student exam scores** using demographic, academic, lifestyle, and institutional features.

The workflow follows strong **machine learning best practices**, including:
- Extensive exploratory data analysis (EDA)
- Clear separation of ordinal and nominal categorical features
- Proper preprocessing with `ColumnTransformer`
- Feature engineering inside pipelines
- Leakage-free cross-validation
- Robust evaluation using multiple regression metrics
- Model persistence for reuse and deployment

The final model is a **Linear Regression** baseline focused on **interpretability and stability**.

---

## Dataset Description
- **Source:** Kaggle (Student Exam Performance Regression Dataset)
- **Files Used:**
  - `train.csv` – Training dataset (includes target)
  - `test.csv` – Test dataset (no target)
  - `sample_submission.csv` – Submission format
- **Target Variable:**  
  - `exam_score` (continuous)

### Feature Groups
- **Demographic:** `age`, `gender`
- **Academic:** `study_hours`, `class_attendance`, `course`, `study_method`
- **Lifestyle:** `sleep_hours`, `sleep_quality`
- **Institutional:** `facility_rating`, `internet_access`
- **Assessment:** `exam_difficulty`

---

## Exploratory Data Analysis (EDA)
Key EDA steps include:
- Distribution analysis for numerical features (age, study hours, sleep hours)
- Boxplots for categorical features vs. exam score
- Regression plots to inspect linear trends
- Category frequency analysis for imbalance detection

Visualizations were created using **Matplotlib** and **Seaborn**.

---

## Feature Engineering & Preprocessing

### Categorical Feature Handling
- **Binary Encoding**
  - `internet_access` → `yes = 1`, `no = 0`
- **Ordinal Encoding**
  - `exam_difficulty`: easy < moderate < hard
  - `facility_rating`: low < medium < high
  - `sleep_quality`: poor < average < good
- **One-Hot Encoding**
  - `study_method`, `course`, `gender`
  - `handle_unknown='ignore'` for robustness

### Numerical Features
- Standardized using `StandardScaler`:
  - `age`
  - `study_hours`
  - `class_attendance`
  - `sleep_hours`

### Engineered Features
Created **after preprocessing** to avoid leakage:
- `quality_sleep_hours`  
  → `sleep_hours × (sleep_quality_encoded + 1)`
- `attended_study_hours`  
  → `study_hours × class_attendance`

All preprocessing and feature engineering are performed **inside pipelines**.

---

## Model Used
- **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)
- Chosen for:
  - Interpretability
  - Strong baseline performance
  - Low variance on large datasets

---

## Evaluation Strategy

### Cross-Validation
- **Method:** 10-Fold Cross-Validation
- **Pipeline:**  
  `Preprocessing → Feature Engineering → Linear Regression`
- Ensures:
  - No data leakage
  - Fair performance estimation

#### Cross-Validation Results
| Metric    | Value        |
|-----------|--------------|
| Mean RMSE | **8.8952**   |
| Std RMSE  | **± 0.0168** |

---

### Test Set Performance
Evaluated on a held-out test split.

| Metric   | Value     |
|----------|-----------|
| RMSE     | **8.700** |
| MAE      | **6.929** |
| R² Score | **0.786** |

These results indicate strong generalization with low variance between CV and test performance.

---

## Visualizations & Interpretability
- Histograms and KDE plots for feature distributions
- Boxplots for categorical feature impact
- Regression plots for linear trend inspection
- Feature engineering designed for interpretability

---

## Tech Stack
- **Python**
- **Libraries**
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
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
