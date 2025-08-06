Heart Disease Prediction Project
Welcome to the Heart Disease Prediction Project, an advanced machine learning endeavor designed to predict heart disease risk with cutting-edge techniques. This project leverages the UCI Heart Disease Dataset to deliver robust predictive models, insightful data analysis, and an interactive user interface for real-time predictions.
Project Overview
This repository encapsulates a comprehensive machine learning pipeline, including data preprocessing, feature engineering, supervised and unsupervised learning, model optimization, and deployment. The project is meticulously organized to ensure reproducibility and ease of use, culminating in a Streamlit-based web application for seamless user interaction.
Project Structure

data/: Stores the cleaned and preprocessed UCI Heart Disease Dataset.
notebooks/: Contains Jupyter notebooks detailing each step of the pipeline:
01_data_preprocessing.ipynb: Data cleaning and preparation.
02_pca_analysis.ipynb: Dimensionality reduction using PCA.
03_feature_selection.ipynb: Feature selection with advanced techniques.
04_supervised_learning.ipynb: Training classification models.
05_unsupervised_learning.ipynb: Clustering analysis.
06_hyperparameter_tuning.ipynb: Model optimization.


models/: Houses the trained machine learning model and encoder (saved as .pkl).
ui/: Source code for the Streamlit web application (app.py).
deployment/: Instructions for deploying the app using Ngrok (ngrok_setup.txt).
results/: Stores evaluation metrics and visualizations.

Installation
Follow these steps to set up the project locally:

Clone the Repository:
git clone https://github.com/YourUsername/Heart_Disease_Project.git
cd Heart_Disease_Project


Install Dependencies:Ensure you have Python 3.8+ installed, then run:
pip install -r requirements.txt



Usage

Run the Notebooks:Execute the Jupyter notebooks in sequence (01_data_preprocessing.ipynb to 06_hyperparameter_tuning.ipynb) to replicate the full pipeline.

Launch the Web Application:Start the Streamlit app for real-time predictions:
streamlit run ui/app.py


Deploy with Ngrok (Optional):Follow the instructions in deployment/ngrok_setup.txt to make the app publicly accessible:

Install Ngrok: pip install pyngrok
Run: ngrok http 8501
Use the generated public URL to access the app.



Dataset
The project utilizes the Heart Disease UCI Dataset, available at:https://archive.ics.uci.edu/dataset/45/heart+disease

Note: The dataset is not included in the repository due to size constraints. Download it from the provided link and place it in the data/ directory, or update the notebook paths accordingly.
Key Features

Data Preprocessing: Handles missing values, encoding, and scaling.
Feature Engineering: Employs PCA and feature selection (RFE, Chi-Square).
Machine Learning Models: Includes Logistic Regression, Decision Trees, Random Forest, SVM, K-Means, and Hierarchical Clustering.
Model Optimization: Utilizes GridSearchCV and RandomizedSearchCV.
Interactive UI: A Streamlit app for user-friendly predictions and visualizations.
Deployment: Ngrok integration for public access.

Contributing
Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your enhancements.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built with Python, Pandas, Scikit-learn, Streamlit, and Ngrok.
Special thanks to the UCI Machine Learning Repository for providing the dataset.
