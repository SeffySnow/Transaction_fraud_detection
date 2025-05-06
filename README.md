

# Transaction Fraud Detection with XGBoost & SageMaker

## 🚀 Overview

A hands‑on end‑to‑end demonstration of detecting credit‑card fraud using:

* Python (pandas, scikit‑learn, XGBoost)
* Exploratory Data Analysis & feature engineering
* Model training & evaluation (XGB, RF, LR)
* AWS SageMaker for scalable training, batch inference, and real‑time endpoint deployment

## 📊 Dataset

* **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (anonymous PCA‑transformed features + Time & Amount)
* **Size:** \~284 K transactions with 492 frauds (0.17 % positive)

## 🔍 Exploratory Data Analysis

* Distribution of frauds by hour of day
* Log‑transform and scaling of the `Amount` feature
* Visual comparisons of `ScaledAmount` and `HourScaled`

## 🛠️ Feature Engineering

* Extract transaction hour from `Time`
* `log1p` transform on `Amount`
* Standard scaling of `Amount` and `Hour`
* Drop raw `Time` & `Amount` after scaling

## 🤖 Modeling

* **Train/Test split:** 70 / 30
* **Algorithms compared:**

  * XGBoost (`scale_pos_weight`)
  * Random Forest (balanced class weights)
  * Logistic Regression (with SMOTE oversampling)
* **Key metrics:**

  * Confusion matrix
  * Precision, recall, F₁‑score
  * ROC AUC & Average Precision (AUPRC)


## 📈 Results

* **Best ROC AUC:** \~0.98
* **Best AUPRC:** \~0.84
* High recall on minority class (\~0.88) with moderate precision


