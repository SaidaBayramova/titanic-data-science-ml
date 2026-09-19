# 🚢 Titanic - End-to-End Data Science & Machine Learning Project

This project covers a complete Data Science pipeline on the classic Titanic dataset, ranging from Exploratory Data Analysis (EDA) and Feature Engineering to Machine Learning tuning, Deep Learning (Keras MLP), Error Analysis, and a live Streamlit Web Application.

---

## 📌 Project Overview
- **Problem Type:** Supervised Learning (Binary Classification)
- **Target Variable:** `Survived` (0 = Did not survive, 1 = Survived)
- **Best Model Accuracy:** **82.12%** (Keras Multi-Layer Perceptron) & **81.56%** (Random Forest / XGBoost)

---

## 🛠️ Data Pipeline & Workflow

1. **Data Cleaning & Imputation:**
   - Filled missing `Age` values with the median.
   - Filled missing `Embarked` values with the mode.
   - Dropped `Cabin` due to high missing rate (~77%).

2. **Feature Engineering & Encoding:**
   - Created `FamilySize` (`SibSp` + `Parch` + 1).
   - Encoded `Sex` (Binary) and `Embarked` (One-Hot Encoding).
   - Normalized numerical features using `StandardScaler` for Deep Learning.

3. **Model Evaluation & Comparison:**

| Model | Accuracy (%) |
| :--- | :--- |
| **Keras Neural Network (MLP)** | **82.12%** |
| **XGBoost Classifier** | **81.56%** |
| **Random Forest (Tuned)** | **81.56%** |
| **LightGBM Classifier** | **81.00%** |
| **Logistic Regression** | **80.45%** |

4. **Error Analysis:**
   - Evaluated False Positives (15) and False Negatives (18).
   - Main finding: The model tends to misclassify male survivors in 3rd class due to strong historical class bias.

---

## 🚀 Interactive Streamlit Web App
The project includes a built-in Streamlit interface allowing users to input custom passenger parameters (Age, Class, Sex, Fare) and get real-time survival predictions with probability scores.

```bash
# Run the Streamlit app locally
streamlit run app.py
