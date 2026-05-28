# ❤️ Cardiovascular Disease Risk Predictor

A machine learning web application that predicts the likelihood of cardiovascular disease based on a patient's clinical and physiological parameters.

---

## 📌 Project Overview

Cardiovascular disease is one of the leading causes of death worldwide. Early detection can save lives. This project applies a full data science pipeline — from raw data cleaning to model deployment — to predict whether a patient is at risk of cardiovascular disease.

---

## 📂 Project Structure

```
├── app.py                  # Streamlit web application
├── nootbook.ipynb          # Full data science notebook (EDA, training, evaluation)
├── cardio_train.csv        # Raw dataset
├── saved_model.pkl         # Trained Random Forest model
├── saved_transformer.pkl   # Fitted StandardScaler transformer
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 📊 Dataset

- **Source:** Cardiovascular Disease Dataset (Kaggle)
- **Size:** ~70,000 patient records, 12 features
- **Target:** `cardio` — 1 (has disease) / 0 (no disease)
- **Features include:** age, gender, height, weight, blood pressure (systolic & diastolic), cholesterol, glucose, smoking, alcohol consumption, physical activity

---

## ⚙️ Project Pipeline

### 1. Data Cleaning
- Removed duplicate rows
- Handled missing/null values (median for numerical, mode for categorical)
- Removed medically impossible blood pressure readings

### 2. Exploratory Data Analysis (EDA)
- Univariate analysis: histograms and boxplots for 6+ variables
- Bivariate analysis: correlation heatmap, scatter plots, count plots
- 6 total visualizations covering all key features

### 3. Feature Engineering
- **BMI** = weight / (height in meters)²
- **Pulse Pressure** = systolic BP − diastolic BP

### 4. Feature Selection
- Used **SelectKBest** with ANOVA F-score (Filter Method) to rank all 13 features by importance

### 5. Model Training — 3 Algorithms Compared

| Model | Precision | Recall |
|---|---|---|
| Logistic Regression | ~0.72 | ~0.72 |
| Support Vector Machine | ~0.73 | ~0.73 |
| **Tuned Random Forest** ✅ | **~0.74** | **~0.74** |

- **Best model:** Random Forest (tuned with GridSearchCV)
- All models achieve precision and recall well above the 0.3 minimum threshold

### 6. Hyperparameter Tuning
- Used `GridSearchCV` on Random Forest with:
  - `n_estimators`: [50, 100]
  - `max_depth`: [None, 10, 20]

### 7. Deployment
- Deployed as an interactive web app using **Streamlit**

---

## 🚀 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/cardio-predictor.git
cd cardio-predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

---

## 🌐 Live Demo

👉 [Click here to open the web app](https://YOUR_STREAMLIT_APP_LINK.streamlit.app)

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Joblib
