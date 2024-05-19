```markdown
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
```

### 2. Data Preprocessing
```python
# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(data[['gender', 'smoking_history']]), columns=encoder.get_feature_names_out(['gender', 'smoking_history']))
data = pd.concat([data.drop(['gender', 'smoking_history'], axis=1), encoded_features], axis=1)
```

### 3. Splitting the Dataset
```python
# Define features and target variable
features = data.drop('diabetes', axis=1)
target = data['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
```

### 4. Training the Decision Tree Model
```python
# Train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
```

### 5. Evaluating the Model
```python
# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
classification_report = metrics.classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report)
```

### 6. Visualizing the Decision Tree
```python
# Visualize the decision tree
dot_data = export_graphviz(clf, out_file=None, feature_names=features.columns, class_names=['Non-Diabetic', 'Diabetic'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("diabetes_decision_tree")
```

### 7. Interactive Prediction Widget
```python
# Define input widgets
age = widgets.FloatText(description="Age:")
gender = widgets.Dropdown(options=['Male', 'Female'], description="Gender:")
bmi = widgets.FloatText(description="BMI:")
hypertension = widgets.Dropdown(options=[0, 1], description="Hypertension:")
heart_disease = widgets.Dropdown(options=[0, 1], description="Heart Disease:")
smoking_history = widgets.Dropdown(options=['never', 'current', 'formerly', 'ever', 'not current'], description="Smoking History:")
HbA1c_level = widgets.FloatText(description="HbA1c Level:")
blood_glucose_level = widgets.FloatText(description="Blood Glucose Level:")
button = widgets.Button(description="Predict")

def predict_diabetes_widget(age, gender, bmi, hypertension, heart_disease, smoking_history, HbA1c_level, blood_glucose_level):
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'bmi': [bmi],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })
    encoded_input_data = pd.DataFrame(encoder.transform(input_data[['gender', 'smoking_history']]), columns=encoder.get_feature_names_out(['gender', 'smoking_history']))
    input_data = pd.concat([input_data.drop(['gender', 'smoking_history'], axis=1), encoded_input_data], axis=1)
    prediction = clf.predict(input_data)
    print("Diabetes Status: ", "Positive" if prediction[0] == 1 else "Negative")

# Display the widget
input_widgets = widgets.VBox([age, gender, bmi, hypertension, heart_disease, smoking_history, HbA1c_level, blood_glucose_level, button])
display(input_widgets)

# Set up button click event
def on_button_click(b):
    predict_diabetes_widget(age.value, gender.value, bmi.value, hypertension.value, heart_disease.value, smoking_history.value, HbA1c_level.value, blood_glucose_level.value)

button.on_click(on_button_click)
```

## Results
### Accuracy
The model achieved an accuracy of 95.24%, indicating that it correctly predicts the diabetes status for 95.24% of the test cases.

### Classification Report
```
              precision    recall  f1-score   support

           0       0.98      0.97      0.97     27453
           1       0.71      0.74      0.72      2547

    accuracy                           0.95     30000
   macro avg       0.84      0.85      0.85     30000
weighted avg       0.95      0.95      0.95     30000
```

### Understanding the Outputs
- **Precision**: Indicates the proportion of true positive predictions. For non-diabetic cases, the precision is 0.98, while for diabetic cases, it is 0.71.
- **Recall**: Indicates the proportion of actual positive cases correctly identified. For non-diabetic cases, the recall is 0.97, while for diabetic cases, it is 0.74.
- **F1-score**: The harmonic mean of precision and recall. For non-diabetic cases, it is 0.97, while for diabetic cases, it is 0.72.
- **Support**: The number of actual occurrences of each class in the test set. There are 27,453 non-diabetic cases and 2,547 diabetic cases.

## Advantages and Disadvantages of Decision Trees
### Advantages
- **Interpretability**: Easy to understand and interpret, making them suitable for medical applications.
- **Non-parametric**: No assumptions about the distribution of data.
- **Handling of Missing Values**: Can handle missing values without significant loss of performance.
- **Feature Importance**: Provides insights into the importance of features in the decision-making process.

### Disadvantages
- **Overfitting**: Prone to overfitting, especially with noisy data.
- **Bias-Variance Tradeoff**: Can suffer from high variance if not properly pruned.
- **Instability**: Small changes in the data can lead to significant changes in the tree structure.

### Comparison with Other Methods
- **Random Forests**: Improve accuracy and reduce overfitting by averaging multiple decision trees, but less interpretable.
- **Support Vector Machines (SVM)**: Handle high-dimensional data well but are less interpretable.
- **Neural Networks**: Capture complex relationships but require large datasets and are computationally intensive.

### Why Decision Trees are Good for Diagnosis
Decision trees offer a balance between accuracy and interpretability, making them suitable for initial diagnostic tools. Their ability to handle various data types and missing values enhances their applicability in medical datasets.

## Conclusion
This project demonstrates the potential of decision trees as a diagnostic tool for diabetes. The model achieved high accuracy, making it a reliable tool for medical professionals. Future improvements could include the use of ensemble methods and techniques to address
