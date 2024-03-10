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
    errors = []
    epochs = 0
    while True:
        total_error = 0
        for inputs, target in zip(training_inputs, training_outputs):
            prediction = perceptron(inputs, weights)
            error = target - prediction
            total_error += error ** 2
            weights[1:] += learning_rate * error * inputs
            weights[0] += learning_rate * error
        errors.append(total_error)
        epochs += 1
        if total_error <= convergence_error or epochs >= max_epochs:
            break
    return errors, epochs

# Plotting epochs against error values
def plot_error(errors, epochs):
    plt.plot(range(1, epochs + 1), errors)
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Epochs vs Error')
    plt.show()

def main():
    # XOR gate truth table inputs and outputs
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_outputs = np.array([0, 1, 1, 0])

    # Initial weights
    initial_weights = np.array([10, 0.2, -0.75])

    # Learning rate
    learning_rate = 0.05

    # Train the perceptron
    initial_errors, num_epochs = train_perceptron(training_inputs, training_outputs, initial_weights, learning_rate)

    # Plot the errors
    plot_error(initial_errors, num_epochs)

    print("Converged in", num_epochs, "epochs")

if __name__ == "__main__":
    main()


