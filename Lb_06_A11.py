# Import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("C:\\Users\\Rishu Jaiswal\\Downloads\\archive (2)\\Data\\features_30_sec.csv")

# Features (X) and target variable (y)
X = df.iloc[:,1:-2]
y = df['label']

# Preprocess Data
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Instantiate MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', random_state=42)

# 6. Train the Model
mlp_classifier.fit(X_train, y_train)

# 7. Evaluate the Model
y_pred = mlp_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)






