import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Loading the dataset
df = pd.read_csv("C:\\Users\\Rishu Jaiswal\\Downloads\\archive (2)\\Data\\features_30_sec.csv")

# DataFrame for country class
country_class_df = df[df['label'] == 'country']

# DataFrame for disco class
classical_class_df = df[df['label'] == 'disco']

# New dataset with two classes
new_data = pd.concat([country_class_df, classical_class_df], axis=0)

# Features (X) and target variable (y)
X = new_data[['spectral_centroid_mean', 'spectral_bandwidth_mean']]
y = new_data['label']

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Function to calculate metrics
def calculate_metrics(y_train, y_test, y_train_pred, y_test_pred):
    
     # Confusion matrix for training data
    cm_train = confusion_matrix(y_train, y_train_pred)

    # Confusion matrix for testing data
    cm_test = confusion_matrix(y_test, y_test_pred)

    # Calculate precision, recall, and F1-score for training data
    precision_train = precision_score(y_train, y_train_pred, pos_label='country')
    recall_train = recall_score(y_train, y_train_pred, pos_label='country')
    f1_train = f1_score(y_train, y_train_pred, pos_label='country')

    # Calculate precision, recall, and F1-score for testing data
    precision_test = precision_score(y_test, y_test_pred, pos_label='country')
    recall_test = recall_score(y_test, y_test_pred, pos_label='country')
    f1_test = f1_score(y_test, y_test_pred, pos_label='country')

    # Calculate accuracy for training and testing data
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    return cm_train, cm_test, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, accuracy_train, accuracy_test

# Model training and prediction
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Function call
cm_train, cm_test, precision_train, recall_train, f1_train, precision_test, recall_test, f1_test, accuracy_train, accuracy_test = calculate_metrics(y_train, y_test, y_train_pred, y_test_pred)

# Print the results
print("Training Data:")
print("Confusion Matrix:")
print(cm_train)
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1-score:", f1_train)
print("Accuracy:", accuracy_train)

print("\nTesting Data:")
print("Confusion Matrix:")
print(cm_test)
print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1-score:", f1_test)
print("Accuracy:", accuracy_test)


# Print the confusion matrices with class labels
print("\n\nTraining Data Confusion Matrix:")
print(pd.DataFrame(cm_train, index=['actual_country', 'actual_disco'], columns=['predicted_country', 'predicted_disco']))

print("\n\nTesting Data Confusion Matrix:")
print(pd.DataFrame(cm_test, index=['actual_country', 'actual_disco'], columns=['predicted_country', 'predicted_disco']))

def model_learning_outcome(accuracy_train, accuracy_test, precision_train, precision_test, recall_train, recall_test, f1_train, f1_test):
    if accuracy_train < accuracy_test:
        if precision_train < precision_test and recall_train < recall_test and f1_train < f1_test:
            return "Regular Fit"
        else:
            return "Overfit"
    else:
        return "Underfit"

# Call the function
learning_outcome = model_learning_outcome(accuracy_train, accuracy_test, precision_train, precision_test, recall_train, recall_test, f1_train, f1_test)
print("Model Learning Outcome:", learning_outcome)


