import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    # Sigmoid activation function.
    return 1 / (1 + np.exp(-x))

def preprocess_data(data):
    # Preprocess the data by shuffling and splitting it into features and labels.
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    # Add bias term to features
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    return X, y

def calculate_weights(X, y):
    # Calculate weights using pseudo-inverse.
    return np.dot(np.linalg.pinv(X), y)

def predict(X, weights):
    # Make predictions using the obtained weights.
    return np.round(sigmoid(np.dot(X, weights))).astype(int)

def calculate_accuracy(predictions, y):
    # Calculate accuracy by comparing predictions with actual labels.
    return np.mean(predictions == y)

def main():
    # Customer data
    data = np.array([
        [20, 6, 2, 386, 1],
        [16, 3, 6, 289, 1],
        [27, 6, 2, 393, 1],
        [19, 1, 2, 110, 0],
        [24, 4, 2, 280, 1],
        [22, 1, 5, 167, 0],
        [15, 4, 2, 271, 1],
        [18, 4, 2, 274, 1],
        [21, 1, 4, 148, 0],
        [16, 2, 4, 198, 0]
    ])

    # Preprocess the data
    X, y = preprocess_data(data)

    # Calculate weights
    weights = calculate_weights(X, y)

    # Make predictions
    predictions = predict(X, weights)

    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, y)
    print("Accuracy with matrix pseudo-inverse:", accuracy)

if __name__ == "__main__":
    main()
