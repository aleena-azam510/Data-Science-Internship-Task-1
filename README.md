# Data-Science-Internship-Tasks

# Project Overview
This repository contains four machine learning tasks covering exploratory data analysis, sentiment analysis, fraud detection, and house price prediction. Each task includes data preprocessing, model implementation, and evaluation.

## Project Steps

### Task 1: EDA and Visualization of a Real-World Dataset
**Description:** Perform exploratory data analysis (EDA) on datasets like the Titanic Dataset or Airbnb Listings Dataset.

#### Steps:
1. **Load the Dataset:** Use Pandas to explore the dataset.
2. **Data Cleaning:**
   - Handle missing values using imputation or removal.
   - Remove duplicates.
   - Identify and manage outliers.
3. **Visualizations:**
   - Bar charts for categorical variables.
   - Histograms for numerical distributions.
   - Correlation heatmap.
4. **Insights:** Summarize key findings.



### Task 2: Text Sentiment Analysis
**Description:** Develop a sentiment analysis model using datasets like IMDB Reviews.

#### Steps:
1. **Text Preprocessing:**
   - Tokenization.
   - Stopword removal.
   - Lemmatization.
2. **Feature Engineering:**
   - Convert text to numerical format using TF-IDF or word embeddings.
3. **Model Training:**
   - Train classifiers such as Logistic Regression or Naive Bayes.
4. **Model Evaluation:**
   - Evaluate performance using precision, recall, and F1-score.



### Task 3: Fraud Detection System
**Description:** Develop a fraud detection model using datasets like the Credit Card Fraud Dataset.

#### Steps:
1. **Data Preprocessing:**
   - Handle imbalanced data using SMOTE or undersampling.
2. **Model Training:**
   - Train models like Random Forest or Gradient Boosting.
3. **Model Evaluation:**
   - Measure precision, recall, and F1-score.
4. **Testing Interface:**
   - Implement a command-line interface for testing.



### Task 4: Predicting House Prices Using the Boston Housing Dataset
**Description:** Implement regression models from scratch to predict house prices.

#### Steps:
1. **Data Preprocessing:**
   - Normalize numerical features.
   - Process categorical variables.
2. **Model Implementation:**
   - Implement Linear Regression, Random Forest, and XGBoost from scratch.
3. **Performance Comparison:**
   - Evaluate models using RMSE and R² metrics.
4. **Feature Importance:**
   - Visualize feature importance for tree-based models.



## How to Run the Scripts
1. Clone this repository:
   
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   
2. Install dependencies:

   pip install -r requirements.txt
   
3. Run each task separately using Jupyter Notebook:
   
   jupyter notebook "1.Task 1 .ipynb"  # For Task 1
   
   jupyter notebook "2.Task 2.ipynb"  # For Task 2
   
   jupyter notebook "3.Task 3.ipynb"  # For Task 3
   
   jupyter notebook "4.Task 4.ipynb"  # For Task 4
   

## Observations
- **EDA:**
  - Identifies patterns and anomalies in data.
  - Helps understand feature relationships through visualizations like heatmaps and histograms.
  - Detects missing values and outliers for better data preprocessing.
  
- **Sentiment Analysis:**
  - Text preprocessing (e.g., stopword removal, lemmatization) significantly impacts model accuracy.
  - TF-IDF and word embeddings enhance feature representation for better classification.
  - Logistic Regression and Naive Bayes perform well on structured textual data.
  
- **Fraud Detection:**
  - Fraudulent transactions are rare, requiring techniques like SMOTE or undersampling to balance the dataset.
  - Precision and recall are crucial since false negatives (missed fraud) can be costly.
  - Tree-based models like Random Forest and Gradient Boosting effectively detect fraud patterns.
  
- **House Price Prediction:**
  - Normalization of numerical features improves model convergence.
  - Linear Regression provides a baseline but struggles with complex relationships.
  - Tree-based models (Random Forest, XGBoost) capture non-linear dependencies, improving predictive accuracy.
  - Feature importance analysis helps identify key factors influencing house prices.


