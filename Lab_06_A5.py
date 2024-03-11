import numpy as np

def sigmoid(x):
    # Sigmoid activation function.
    return 1 / (1 + np.exp(-x))

class Perceptron:
    def __init__(self, input_size):
        # Initialize the perceptron with random weights and bias.
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = 0.01  # Learning rate

    def train(self, X, y, epochs):
        # Train the perceptron using gradient descent.
        for epoch in range(epochs):
            # Iterate through each data point
            for i in range(X.shape[0]):
                # Forward pass
                z = np.dot(X[i], self.weights) + self.bias
                a = sigmoid(z)

                # Compute gradients for weight and bias
                dcost_dz = a - y[i]
                dz_dw = X[i]
                dz_db = 1

                # Update weights and bias using gradient descent
                self.weights -= self.lr * dcost_dz * dz_dw
                self.bias -= self.lr * dcost_dz * dz_db

    def predict(self, X):
        # Make predictions using the trained perceptron.
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

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

    # Shuffle data
    np.random.shuffle(data)

    # Split features and labels
    X = data[:, :-1]
    y = data[:, -1]

    # Initialize and train the perceptron
    input_size = X.shape[1]
    perceptron = Perceptron(input_size)
    perceptron.train(X, y, epochs=1000)

    # Test the trained perceptron
    predictions = perceptron.predict(X)
    predictions = np.round(predictions).astype(int)

    # Compare predictions with actual labels to compute accuracy
    accuracy = np.mean(predictions == y)
    print("# Accuracy:", accuracy)

if __name__ == "__main__":
    main()
