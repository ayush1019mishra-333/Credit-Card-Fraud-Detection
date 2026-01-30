# Credit Card Fraud Detection

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)

## Overview

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used is from Kaggle and contains various features related to credit card transactions, with the task of classifying each transaction as either fraudulent or legitimate.

## Dataset

HEAD
The dataset contains features of credit card transactions. Due to confidentiality, the variables are hidden. The target variable (`Class`) represents the transaction type: `0` for normal transactions and `1` for fraudulent transactions.

The dataset contains anonymized features of credit card transactions. Due to confidentiality, the variables are hidden. The target variable (`Class`) represents the transaction type: `0` for normal transactions and `1` for fraudulent transactions.
 3b257be (Uploading project files)

## Preprocessing

The following steps were performed to prepare the dataset:

1. **Standardization**: The `Amount` column was standardized to improve model performance.
2. **Data Cleaning**: Duplicate rows were removed, and the `Time` column was dropped due to its irrelevance to the classification task.
3. **Class Distribution**: An imbalance between the classes (fraud vs. normal) was observed, leading to the application of class-balancing techniques.

## Modeling

To address the class imbalance, the following approaches were applied:

1. **Downsampling**: The normal class was downsampled to match the number of fraud transactions.
2. **Oversampling (SMOTE)**: Synthetic samples were generated for the fraud class to balance the dataset.

Three machine learning models were tested:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Decision Tree Classifier**

Each classifier was trained on three different versions of the dataset: the original, downsampled, and oversampled datasets.

## Evaluation

The models were evaluated based on the following metrics:

- **Accuracy**: Proportion of correctly predicted transactions.
HEAD
- **F1 Score**: The harmonic mean of precision and recall.
- **Recall**: The proportion of actual fraudulent transactions correctly identified.
- **Precision**: The proportion of true positive fraud predictions out of all positive predictions.
- **Accuracy**: The proportion of actual fraudulent transactions precisely checked.

It’s worth noting that when trained on the original, imbalanced dataset, the models displayed a high accuracy but low precision, indicating that they were predominantly predicting the majority class (normal transactions). This demonstrates how class imbalance can create a false sense of performance, as the model may appear effective due to high accuracy but fail to identify the minority class (fraudulent transactions) adequately. Therefore, class-balancing techniques like SMOTE and downsampling were crucial to improving model performance, particularly for the detection of fraudulent transactions.

## Results

Each model was trained and tested on the original, downsampled, and oversampled datasets. The performance metrics for each model were recorded, and the model with the highest F1 score was selected as the best-performing model.

## Conclusion

 HEAD
This project underscores that accuracy is a deceptive metric in the context of fraud detection, where the "needle in the haystack" is what truly matters. By transitioning from a skewed dataset to a balanced one using SMOTE and downsampling, we shifted the model’s focus from simple pattern recognition of normal behavior to the surgical identification of fraudulent anomalies. This recalibration successfully bridged the gap between high theoretical scores and actual operational utility, significantly boosting precision and recall. Ultimately, the results prove that a robust fraud detection system must prioritize class-specific sensitivity over aggregate accuracy to minimize the financial and reputational costs of undetected scams.
