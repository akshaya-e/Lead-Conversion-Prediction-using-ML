LEAD CONVERSION PREDICTION USING MACHINE LEARNING

OVERVIEW

A machine learning classification pipeline built in Python to predict whether
prospective sales leads will convert into paying customers. This system helps
sales and marketing teams identify high-potential leads, prioritize follow-ups,
and optimize campaign strategies.


DATASET

Table: Unified_Leads.csv
Historical Lead Records | Binary Classification Target (Converted vs. Not Converted)

Lead columns:        Name, contact_number, Contact_email (Anonymized PII)
Source columns:      source_type, campaign_name, references
Sales columns:       lead_stage, assigned_to
Interaction columns: creation_date, first_call_attempt, last_call_date, last_call_time

Target Variable:     target (1 = CONVERTED, 0 = NOT CONVERTED)


PREPROCESSING & FEATURE ENGINEERING

Data Cleaning        Removed anonymized PII columns and 100% null columns
Missing Imputation   Median imputation for numerical features; constant ('missing')
                     imputation for categorical variables
Encoding             One-Hot Encoding (`handle_unknown='ignore'`) for categorical data
Feature Scaling      Standardization (`StandardScaler`) applied to numerical attributes
Date Extraction      Extracted `creation_year`, `creation_month`, `creation_day`,
                     and `creation_hour` from `creation_date`

MODEL PIPELINE & ARCHITECTURE

TRAIN-TEST SPLIT
  - 80% Training set / 20% Testing set 

CLASSIFICATION MODELS EVALUATED
  - Logistic Regression 
  - Decision Tree Classifier
  - Random Forest Classifier 

EVALUATION METRICS
  - Accuracy, Precision, Recall, F1-Score, and ROC-AUC Score


PROJECT WORKFLOW STEPS

1. Data Preprocessing & Target Definition
   - Mapped `lead_stage == 'CONVERTED'` to binary target `1` and all other stages to `0`.
   - Dropped raw timestamps after extracting temporal features.

2. Exploratory Data Analysis (EDA)
   - Visualized target class distribution.
   - Analyzed conversion probability across different `source_type` channels.

3. Model Development & Evaluation
   - Built unified `ColumnTransformer` and `Pipeline` workflows to prevent data leakage.
   - Benchmarked multiple classifiers across key binary classification metrics.

4. Prediction & Lead Prioritization System
   - Developed `predict_new_lead()` to classify incoming leads and compute exact
     conversion probabilities 
   - Ranked test leads from highest to lowest conversion probability to guide sales
     team outreach.


