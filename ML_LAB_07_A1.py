import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Load the dataset
df = pd.read_csv("C:\\Users\\Rishu Jaiswal\\Downloads\\archive (2)\\Data\\features_30_sec.csv")

# Features (X) and target variable (y)
X = df.iloc[:,1:-2]
class_column = df['label']  

# Label encode the class column
le = LabelEncoder()
df['class_column'] = le.fit_transform(class_column)

y = df['class_column']

# Preprocess Data
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define Hyperparameter Grids
perceptron_grid = {
    'tol': [1e-3, 1e-4, 1e-5],
    'eta0': [0.01, 0.1, 1.0],
    'max_iter': [100, 200, 500]
}

mlp_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
    'solver': ['adam', 'sgd']
}

# Define Estimators
perceptron = Perceptron(random_state=42)
mlp = MLPRegressor(random_state=42)

# Create RandomizedSearchCV Objects (adjust 'scoring' for classification)
perceptron_cv = RandomizedSearchCV(perceptron, perceptron_grid, cv=5, scoring='neg_mean_squared_error', n_iter=10)
mlp_cv = RandomizedSearchCV(mlp, mlp_grid, cv=5, scoring='neg_mean_squared_error', n_iter=10)

# Train the Search
perceptron_cv.fit(X_train, y_train)
mlp_cv.fit(X_train, y_train)

# Access Best Estimators and Parameters
best_perceptron = perceptron_cv.best_estimator_
best_mlp = mlp_cv.best_estimator_

best_perceptron_params = perceptron_cv.best_params_
best_mlp_params = mlp_cv.best_params_

# Evaluate Performance
perceptron_pred = best_perceptron.predict(X_test)
mlp_pred = best_mlp.predict(X_test)

perceptron_mse = mean_squared_error(y_test, perceptron_pred)
mlp_mse = mean_squared_error(y_test, mlp_pred)

print(f"Perceptron MSE: {perceptron_mse:.4f}")
print(f"MLP MSE: {mlp_mse:.4f}")
print(f"Perceptron Best Parameters: {best_perceptron_params}")
print(f"MLP Best Parameters: {best_mlp_params}")
