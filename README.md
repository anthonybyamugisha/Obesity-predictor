# 🧠 Obesity Level Prediction — Explainable Machine Learning

> 
> **Dataset:** [Obesity Dataset — Hugging Face](https://huggingface.co/datasets/aiml2021/obesity)

## 📌 Project Overview

This project trains and interprets three machine learning models to predict obesity levels based on lifestyle and physical attributes. The focus is not just on accuracy but on **understanding why** each model makes its decisions, using two explainability techniques: **SHAP** and **LIME**.

---

## 📂 Project Structure

```
📦 GroupM_ML_Coursework
 ┣ 📓 GroupM_ML_third_course_work.ipynb   # Main notebook
 ┣ 📄 README.md                           # This file
```

---

## 📊 Dataset

The **Obesity Dataset** contains 2,111 records and 17 features covering:

- **Physical attributes** — Age, Height, Weight
- **Dietary habits** — FAVC, FCVC, NCP, CAEC, CALC, CH2O
- **Lifestyle** — FAF (physical activity), TUE (tech use), SMOKE, SCC
- **Transport** — MTRANS
- **Target** — `NObeyesdad` (7 obesity classes)

No missing values were found in the dataset.

---

## ⚙️ ML Workflow

### 1. Exploratory Data Analysis
- Distribution of obesity levels
- Gender vs obesity level breakdown
- Age distribution
- Correlation heatmap of numeric features

### 2. Preprocessing
- Dropped target column to separate features and labels
- One-hot encoded categorical features (`pd.get_dummies`)
- Label-encoded target variable
- 80/20 train-test split (`random_state=42`)
- Applied `StandardScaler` (used for Neural Network only)

### 3. Models Trained

| Model | Configuration |
|-------|--------------|
| Decision Tree | `max_depth=5`, `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42` |
| Neural Network (MLP) | `hidden_layer_sizes=(64,32)`, `max_iter=500` |

---

## 📈 Results

| Model | Accuracy |
|-------|----------|
| Decision Tree | 83.5% |
| Neural Network (MLP) | 94.1% |
| **Random Forest** | **95.5% ✅** |

The Random Forest achieved the best performance, with near-perfect scores across all 7 obesity classes.

---

## 🔍 Model Explainability

### SHAP (SHapley Additive exPlanations)
- `TreeExplainer` used for Decision Tree and Random Forest
- `KernelExplainer` used for Neural Network
- SHAP interaction value plots generated for all three models

**Key finding:** All three models agree that `Weight` is the most important feature. However, the Decision Tree relies on it almost exclusively, the Random Forest adds Age and their interaction, and the Neural Network additionally captures Height and dietary habit interactions (FCVC, NCP).

### LIME (Local Interpretable Model-agnostic Explanations)
- `LimeTabularExplainer` used across all three models
- Single-instance explanations generated per model

**Key finding:** The Decision Tree's LIME explanation is dominated by a single weight threshold (`Weight <= 66`). The Random Forest spreads evidence more evenly across multiple features. The Neural Network LIME requires the scaler to be embedded inside the prediction function — passing raw unscaled inputs produces meaningless all-zero explanations.

---

## 🛠️ Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap lime
```

---

## 🚀 How to Run

1. Clone this repository
```bash
git clone https://github.com/anthonybyamugisha/Obesity-predictor.git
cd GroupM-ML-Coursework
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Open the notebook
```bash
jupyter notebook GroupM_ML_third_course_work.ipynb
```

4. Run all cells from top to bottom

---

## 📝 Key Takeaways

- SHAP is better for understanding a model's overall behaviour across the whole dataset
- LIME is better for explaining one specific prediction to a non-technical audience
- Neural networks are the hardest to explain — both SHAP and LIME use approximations, and preprocessing must be handled carefully
- The accuracy-interpretability trade-off is real: the Decision Tree is the easiest to explain but the least accurate

---