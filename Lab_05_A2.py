import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the Excel file
file_path = "C:/Users/Rishu Jaiswal/Downloads/Lab Session1 Data.xlsx"
purchase_data = pd.read_excel(file_path, sheet_name="Purchase data")

# Segregate data into matrices A and C
A = purchase_data.iloc[:, 1:4]  
C = purchase_data.iloc[:, 4]

# Calculate the pseudo-inverse of A
A_inv = np.linalg.pinv(A)

# Calculate the model vector X
X = np.dot(A_inv, C)

# Calculate evaluation metrics
mse = mean_squared_error(C, np.dot(A, X))  
rmse = np.sqrt(mse)
mape = np.mean(np.abs((C - np.dot(A, X)) / C)) * 100  
r2 = r2_score(C, np.dot(A, X))  # Use predicted values from A*X instead of X directly

# Print evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("R-squared (R2) Score:", r2)
