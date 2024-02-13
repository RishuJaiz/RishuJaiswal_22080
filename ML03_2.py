import pandas as pd
import numpy as np

# Load the data from the Excel file
file_path = "C:/Users/Rishu Jaiswal/Downloads/Lab Session1 Data.xlsx"
purchase_data = pd.read_excel(file_path, sheet_name="Purchase data")

# Segregate data into matrices A and C
A = purchase_data.iloc[:,1 :4]  
C = purchase_data.iloc[:,4 :5]   



# Dimensionality of the vector space
dimensionality = A.shape[1]

# Number of vectors in the vector space
num_vectors = A.shape[0]

# Rank of matrix A
rank_A = np.linalg.matrix_rank(A)

# Using Pseudo-Inverse to find the cost of each product
pseudo_inverse_A = np.linalg.pinv(A)
cost_per_product = np.dot(pseudo_inverse_A, C)

# Print results
print("Dimensionality of the vector space:", dimensionality)
print("Number of vectors in the vector space:", num_vectors)
print("Rank of Matrix A:", rank_A)
print("Cost of each product available for sale:")
print(cost_per_product)

# Calculate the pseudo-inverse of A
A_inv = np.linalg.pinv(A)

# Calculate the model vector X
X = np.dot(A_inv, C)

print("Model vector X:")
print(X)


