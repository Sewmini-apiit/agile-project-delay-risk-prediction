# Predictive Analytics for Project Delay Risks in Agile Software Development

This repository contains the source code developed for my MSc Computer Science dissertation.

## Research

The study investigates the use of supervised machine learning to identify elevated project delay-risk exposure in Agile software development projects.

## Dataset

The analysis uses the Project Management Risk Raw dataset obtained from Kaggle. The dataset is synthetic, as stated by the dataset author.

The primary analysis filters the dataset to IT projects using Agile, Scrum and Kanban methodologies.

## Models

Three supervised machine-learning models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The analysis also includes model evaluation, threshold analysis, feature interpretation, SHAP analysis, sensitivity testing and multiclass classification.

## Source Code

The complete analysis is available in:

`Agile_Project_Delay_Risk_Analysis.ipynb`

## Important Limitation

The dataset uses Risk_Level as the target rather than an observed delayed/on-time delivery outcome. Therefore, the study evaluates elevated project risk exposure as a proxy and should not be interpreted as a production-ready delay prediction system.
