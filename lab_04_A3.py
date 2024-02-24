import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Rishu Jaiswal\\Downloads\\archive (2)\\Data\\features_30_sec.csv")

# Function to calculate Minkowski distance
def minkowski_distance(x, y, r):
    a  = np.sum(np.abs(x - y) ** r) ** (1 / r)
    return a

# two feature vectors from dataset
X1 = data["chroma_stft_mean"]
X2 = data["rms_mean"]

# Calculate Minkowski distance for r value from 1 to 10
r_values = range(1,11)
print(r_values)
r_values
distances = [minkowski_distance(X1, X2, r) for r in r_values]

# Plot the distances
plt.plot(r_values, distances, marker='*')
plt.xlabel('Value of r')
plt.ylabel('Minkowski Distance')
plt.title('Minkowski Distance vs. Value of r')
plt.grid(True)
plt.show()
