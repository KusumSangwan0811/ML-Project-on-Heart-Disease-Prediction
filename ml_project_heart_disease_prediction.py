# -*- coding: utf-8 -*-

# Importing required libraries
import seaborn as sns
#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as ex

# Load the dataset
heart_df = pd.read_csv('heart.csv')

heart_df.head()

heart_df.tail()


# Checking the data shape
heart_df.shape

# Checking the number of distinct entries in each column
print(heart_df.nunique())

heart_df.info()

heart_df.describe()

# Checking for missing values
print(heart_df.isnull().sum())


# Chceking for duplicates
heart_df[heart_df.duplicated()]

#This is to look at what all unique values have.
list_col=['sex','chol','trtbps','cp','thall','exng']

for col in list_col:
    print('{} :{} ' . format(col.upper(),heart_df[col].unique()))

ex.pie(heart_df,names='output',title='Proportion of different classes')

# Correlation Matrix
corr_matrix=heart_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the correlation matrix
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)

# Set the title
plt.title('Correlation Heatmap of Heart Disease Dataset')

# Show the plot
plt.show()

# Plotting the distribution of each feature
features = heart_df.columns

# Set the style of the visualization
sns.set(style="whitegrid")

# Plot distribution for each feature
plt.figure(figsize=(20, 15))
for i, feature in enumerate(features):
    plt.subplot(len(features) // 3 + 1, 3, i + 1)
    sns.histplot(heart_df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()

# create a histplot trestbops column to analyse with sex column
sns.histplot(heart_df, x='trtbps', kde=True, palette = "Spectral", hue ='sex')



# Splitting the data into features and output
X = heart_df.drop(columns='output')
y = heart_df['output']

# Loading libraries for model building
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using Standard scaler to standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# List of models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost Classifier": XGBClassifier()
}

# Function to train and evaluate a model
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return accuracy

# Dictionary to store the accuracy of each model
accuracy_dict = {}

# Training and evaluating each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    accuracy = train_and_evaluate(model, X_train, X_test, y_train, y_test)
    accuracy_dict[model_name] = accuracy
    print("\n")


# Create a bar plot for accuracies
plt.figure(figsize=(5,3))
plt.barh(list(accuracy_dict.keys()), list(accuracy_dict.values()), color='skyblue')
plt.xlabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.xlim(0, 1)
plt.show()



# Define the parameter grid for KNN
param_grid = {
    'n_neighbors': list(range(15,25)),  # broader range for n_neighbors
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# Initialize KNN classifier
knn = KNeighborsClassifier()

# Initialize GridSearchCV with StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=cv, n_jobs=-1, scoring='accuracy')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation score: {best_score}")

# Train the best KNN model
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)

# Evaluate the best KNN model
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test set accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save the trained model to a file using Pickle
import pickle
pickle.dump(best_knn, open('knn_model.pkl', 'wb'))

