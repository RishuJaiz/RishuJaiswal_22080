import numpy as np
import matplotlib.pyplot as plt

# Step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Perceptron function
def perceptron(inputs, weights):
    weighted_sum = np.dot(inputs, weights[1:]) + weights[0]
    return step_activation(weighted_sum)

# Initialize weights
def initialize_weights(num_inputs):
    return np.random.uniform(-0.5, 0.5, size=num_inputs + 1)

# Train the perceptron
def train_perceptron(training_inputs, training_outputs, weights, learning_rate, convergence_error=0.002, max_epochs=1000):
    epochs = 0
    while True:
        total_error = 0
        for inputs, target in zip(training_inputs, training_outputs):
            prediction = perceptron(inputs, weights)
            error = target - prediction
            total_error += error ** 2
            weights[1:] += learning_rate * error * inputs
            weights[0] += learning_rate * error
        epochs += 1
        if total_error <= convergence_error or epochs >= max_epochs:
            break
    return epochs

def main():
    # XOR gate data
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([0, 1, 1, 0])

    # Initial weights
    initial_weights = np.array([10, 0.2, -0.75])

    # Learning rates to test
    learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    num_epochs_list = []

    # Train the perceptron for each learning rate
    for learning_rate in learning_rates:
        weights = initial_weights.copy()
        num_epochs = train_perceptron(training_inputs, training_outputs, weights, learning_rate)
        num_epochs_list.append(num_epochs)

    # Plotting
    plt.plot(learning_rates, num_epochs_list, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Number of Epochs')
    plt.title('Number of Epochs vs Learning Rate')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
