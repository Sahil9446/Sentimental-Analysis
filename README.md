# Sentiment Analysis with Machine Learning

## Overview
This project performs sentiment analysis on a dataset using various machine learning models, predicting whether input words or phrases are associated with positive, neutral, or negative sentiments.

## Dataset
The labeled dataset used in this project includes examples of words/phrases with corresponding sentiments (positive, neutral, or negative). The dataset, sourced from Kaggle, ensures a well-labeled collection for accurate model training and evaluation. The dataset can be found [here](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?resource=download).

## Models Used
Several machine learning models were implemented and compared:

1. **Baseline Model:** A simple baseline model to establish performance benchmarks.
2. **Logistic Regression:** A linear model for binary and multiclass classification. The model achieved an accuracy of 83.10%.
3. **Decision Tree:** A non-linear model making decisions based on a tree-like graph.
4. **Random Forest:** An ensemble model of decision trees for enhanced accuracy and generalization.

## Best Performing Model
The Logistic Regression model exhibited the highest accuracy among the evaluated models. While accuracy was 0.8309679767103348, considerations for precision, recall, and F1 score are advisable based on specific project requirements.

![Logistic Regression Accuracy](path/to/Logistic_regression_accuracy.png)

## Usage
To use the sentiment analysis model:

1. Open the provided Jupyter notebook.
2. Execute the notebook cells for dataset loading, preprocessing, and model training.
3. Input a word or phrase when prompted, and the model will predict sentiment as positive, neutral, or negative.

## Requirements
Ensure the following dependencies are installed:

```bash
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, ConfusionMatrixDisplay
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
