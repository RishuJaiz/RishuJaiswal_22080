import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:\\Users\\Rishu Jaiswal\\Downloads\\archive (2)\\Data\\features_30_sec.csv")


# Sample data


X = df['chroma_stft_mean']
y = df['label']
print(X)
print(y)


# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Displaying the results
print("X_train:")
print(X_train)
print("\ny_train:")
print(y_train)
print("\nX_test:")
print(X_test)
print("\ny_test:")
print(y_test)
