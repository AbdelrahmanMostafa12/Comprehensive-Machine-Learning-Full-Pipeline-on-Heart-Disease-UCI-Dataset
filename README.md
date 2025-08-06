
# 💓 Heart Disease Prediction Project

Welcome to the **Heart Disease Prediction Project**, an advanced machine learning endeavor designed to predict heart disease risk using cutting-edge techniques.  
This project leverages the **UCI Heart Disease Dataset** to deliver robust predictive models, insightful data analysis, and an interactive user interface for real-time predictions.

---

## 📌 Project Overview

This repository encapsulates a **comprehensive machine learning pipeline**, including:

- 🔍 Data preprocessing
- 🛠️ Feature engineering
- 📚 Supervised & unsupervised learning
- ⚙️ Model optimization
- 🌐 Deployment with Streamlit and Ngrok

The project is **well-organized** for clarity and reproducibility, and culminates in a user-friendly web app built with Streamlit.

---

## 📁 Project Structure

```
├── data/                    # Cleaned and preprocessed dataset
├── notebooks/               # Jupyter notebooks for each pipeline stage
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
├── models/                  # Saved ML model and encoder (.pkl)
├── ui/                      # Streamlit app source code (app.py)
├── results/                 # Evaluation metrics and visualizations
└── requirements.txt         # Project dependencies
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YourUsername/Heart_Disease_Project.git
cd Heart_Disease_Project
```

### 2️⃣ Install Dependencies
Ensure you have **Python 3.8+** installed:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶️ Run the Notebooks
Execute the Jupyter notebooks **in sequence** to build the pipeline:

```text
01_data_preprocessing.ipynb ➝ 06_hyperparameter_tuning.ipynb
```

### 🌐 Launch the Web Application
Start the Streamlit app for real-time predictions:
```bash
streamlit run ui/app.py
```

### 🌍 Deploy with Ngrok (Optional)
Make the app publicly accessible:

1. Install Ngrok:
   ```bash
   pip install pyngrok
   ```
2. Run Ngrok:
   ```bash
   ngrok http 8501
   ```
3. Use the generated public URL to access the app.

---

## 🧠 Dataset

This project uses the **UCI Heart Disease Dataset**:  
📥 [Download it from the UCI Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)

> ⚠️ **Note:** The dataset is **not included** in the repository due to size constraints.  
Download and place it in the `data/` directory, or update notebook paths as needed.

---

## ✨ Key Features

- ✅ **Data Preprocessing:** Missing values, encoding, and feature scaling  
- 🧬 **Feature Engineering:** PCA, RFE, Chi-Square selection  
- 🤖 **ML Models:** Logistic Regression, Decision Trees, Random Forest, SVM  
- 🔍 **Unsupervised Learning:** K-Means, Hierarchical Clustering  
- 🛠️ **Optimization:** GridSearchCV & RandomizedSearchCV  
- 🖥️ **Interactive UI:** Streamlit app for predictions & visualization  
- 🌐 **Deployment Ready:** Ngrok integration for public access  

---

## 🙏 Acknowledgments

- Built with **Python**, **Pandas**, **Scikit-learn**, **Streamlit**, and **Ngrok**
- Special thanks to the **UCI Machine Learning Repository** for providing the dataset.
