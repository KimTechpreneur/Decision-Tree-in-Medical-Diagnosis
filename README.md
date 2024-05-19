# Diabetes Diagnosis Using Decision Trees

This project demonstrates the use of decision tree classifiers for diagnosing diabetes based on a dataset of medical and demographic features. The implementation is done using Python with the help of Scikit-learn, Pandas, and other libraries.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Understanding the Code](#understanding-the-code)
- [Results](#results)
- [Advantages and Disadvantages of Decision Trees](#advantages-and-disadvantages-of-decision-trees)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
The purpose of this project is to develop a decision tree model to diagnose diabetes. The decision tree model is chosen for its simplicity and interpretability, making it suitable for medical diagnosis applications.

## Dataset
The dataset used for this project is the `diabetes_prediction_dataset.csv`, which includes the following features:
- Age
- Gender
- Body Mass Index (BMI)
- Hypertension
- Heart Disease
- Smoking History
- HbA1c Level
- Blood Glucose Level

The target variable is the diabetes status (positive or negative).

## Requirements
- Python 3.x
- Pandas
- Scikit-learn
- Graphviz
- Ipywidgets
- Google Colab (optional for running the notebook online)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/diabetes-diagnosis.git
    cd diabetes-diagnosis
    ```

2. Install the required packages:
    ```bash
    pip install pandas scikit-learn graphviz ipywidgets
    ```

## Usage
1. **Training and Visualizing the Decision Tree:**
   Run the provided Jupyter Notebook or Python script to train the decision tree model on the dataset and visualize the tree.
   
2. **Interactive Widget for Prediction:**
   The notebook includes an interactive widget to input patient data and predict diabetes status using the trained model.

## Understanding the Code
The code is divided into the following sections:

### 1. Importing Libraries and Loading the Dataset
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
import graphviz
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
data = pd.read_csv('diabetes_prediction_dataset.csv')
