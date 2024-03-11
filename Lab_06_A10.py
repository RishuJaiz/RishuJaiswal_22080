from sklearn.neural_network import MLPClassifier

def train_and_gate():
    # Define the AND Gate dataset
    X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_and = [0, 0, 0, 1]

    # Initialize MLPClassifier for AND Gate
    and_classifier = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000)

    # Train the AND Gate classifier
    and_classifier.fit(X_and, y_and)

    return and_classifier

def train_xor_gate():
    # Define the XOR Gate dataset
    X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_xor = [0, 1, 1, 0]

    # Initialize MLPClassifier for XOR Gate
    xor_classifier = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=1000)

    # Train the XOR Gate classifier
    xor_classifier.fit(X_xor, y_xor)

    return xor_classifier

def main():
    # Train AND Gate classifier
    and_classifier = train_and_gate()

    # Train XOR Gate classifier
    xor_classifier = train_xor_gate()

    # Predictions for AND Gate
    print("AND Gate Predictions:")
    for inputs in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        prediction = and_classifier.predict([inputs])
        print(f"{inputs} -> {prediction}")

    # Predictions for XOR Gate
    print("\nXOR Gate Predictions:")
    for inputs in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        prediction = xor_classifier.predict([inputs])
        print(f"{inputs} -> {prediction}")

if __name__ == "__main__":
    main()
