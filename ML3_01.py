import pandas as pd

file_path = 'C:/Users/Rishu Jaiswal/Downloads/Lab Session1 Data.xlsx'

df = pd.read_excel(file_path)
purchase_data = pd.read_excel(file_path)


# Select the features (columns) you want to include in the matrix
selected_features = ['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']

# Extract the selected features into a matrix
feature_matrix = df[selected_features].values

# Display the feature matrix
A = print(feature_matrix)

C =df['Payment (Rs)'].values

print(C)
