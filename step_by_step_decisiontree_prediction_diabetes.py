# -*- coding: utf-8 -*-
"""Step by step Decisiontree prediction diabetes"""

# Import necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Function to load data
def load_data(file_name):
    if os.path.exists(file_name):
        data = pd.read_csv(file_name)
        print("Data loaded successfully.")
    else:
        print(f"File '{file_name}' not found in the current directory.")
        file_path = input("Please provide the full path to the dataset file: ")
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
    return data

# Load data
file_name = 'diabetes_prediction_dataset.csv'
data = load_data(file_name)

# Display the first few rows of the dataset
print(data.head())

# Define features and target variable
FEATURES = ['age', 'gender', 'bmi', 'hypertension', 'heart_disease', 'smoking_history', 'HbA1c_level', 'blood_glucose_level']
TARGET = 'diabetes'

# Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
encoded_features = pd.DataFrame(encoder.fit_transform(data[['gender', 'smoking_history']]),
                                columns=encoder.get_feature_names_out(['gender', 'smoking_history']))

# Drop the original categorical features and concatenate the encoded features
data = pd.concat([data.drop(['gender', 'smoking_history'], axis=1), encoded_features], axis=1)

# Split the data into training and testing sets
X = data.drop(TARGET, axis=1)
y = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Ensure Graphviz executables are installed
try:
    import graphviz
    dot_data = export_graphviz(clf, out_file=None,
                               feature_names=X.columns,
                               class_names=['Negative', 'Positive'],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("diabetes_decision_tree")
except ImportError:
    print("Graphviz library is not installed. Please install it using the following command:")
    print("pip install graphviz")
except graphviz.backend.ExecutableNotFound:
    print("Graphviz executable not found. Please ensure Graphviz is installed and added to your PATH.")
    print("You can install Graphviz using the following commands:")
    print("Ubuntu/Debian: sudo apt-get install graphviz")
    print("macOS (using Homebrew): brew install graphviz")
    print("Windows: Download from https://graphviz.gitlab.io/_pages/Download/Download_windows.html")

# Function to take user input and make predictions using widgets
def predict_diabetes_widget(age, gender, bmi, hypertension, heart_disease, smoking_history, HbA1c_level, blood_glucose_level):
    # Create DataFrame for input features
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'HbA1c_level': [HbA1c_level],
        'blood_glucose_level': [blood_glucose_level],
        'gender_Female': [1 if gender == 'Female' else 0],
        'gender_Male': [1 if gender == 'Male' else 0],
        'smoking_history_never': [1 if smoking_history == 'never' else 0],
        'smoking_history_current': [1 if smoking_history == 'current' else 0],
        'smoking_history_ex': [1 if smoking_history == 'ex' else 0],
        'smoking_history_currently': [1 if smoking_history == 'currently' else 0],
        'smoking_history_No Info': [1 if smoking_history == 'No Info' else 0],
    })

    # Ensure all feature names are present in input_data and in correct order
    feature_names = X.columns
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0

    # Reorder columns to match the training set
    input_data = input_data[feature_names]

    # Predict
    prediction = clf.predict(input_data)
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    print(f"Diabetes Prediction: {result}")

# Create widgets for input
age_widget = widgets.FloatText(description="Age:")
gender_widget = widgets.Dropdown(options=['Male', 'Female'], description="Gender:")
bmi_widget = widgets.FloatText(description="BMI:")
hypertension_widget = widgets.Dropdown(options=[0, 1], description="Hypertension:")
heart_disease_widget = widgets.Dropdown(options=[0, 1], description="Heart Disease:")
smoking_history_widget = widgets.Dropdown(options=['never', 'current', 'ex', 'currently', 'No Info'], description="Smoking History:")
HbA1c_level_widget = widgets.FloatText(description="HbA1c Level:")
blood_glucose_level_widget = widgets.FloatText(description="Blood Glucose Level:")
button_widget = widgets.Button(description="Predict")

# Display widgets
display(age_widget, gender_widget, bmi_widget, hypertension_widget, heart_disease_widget,
        smoking_history_widget, HbA1c_level_widget, blood_glucose_level_widget, button_widget)

# Set up button click event
def on_button_click(b):
    predict_diabetes_widget
