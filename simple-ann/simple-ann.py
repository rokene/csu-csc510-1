#!/usr/bin/env python3

import numpy as np

# Activation function: Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is the output of the sigmoid function
    return x * (1 - x)

# Generate dataset
# For simplicity, we'll use the series: 1, 2, 3, ..., 10
# Input: [1,2] Output: 3
# Input: [2,3] Output: 4
# ...
# Input: [8,9] Output: 10

def create_dataset(series, window_size):
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# Define the series
series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create input-output pairs
window_size = 2
X, y = create_dataset(series, window_size)

# Normalize the data (optional, helps with training)
# Here, since the series is small and linear, normalization isn't strictly necessary
# But it's a good practice
X = X / float(np.max(series))
y = y / float(np.max(series))

# Define network architecture
input_size = window_size    # 2
hidden_size = 2             # Number of neurons in hidden layer
output_size = 1             # 1 (predicting the next number)

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(input_size, hidden_size)  # Weights from Input to Hidden layer
b1 = np.random.randn(1, hidden_size)           # Bias for Hidden layer
W2 = np.random.randn(hidden_size, output_size) # Weights from Hidden to Output layer
b2 = np.random.randn(1, output_size)           # Bias for Output layer

# Define hyperparameters
learning_rate = 0.1
epochs = 1000

# Training loop
for epoch in range(epochs):
    # ----- Forward Propagation -----
    # Input to Hidden
    z1 = np.dot(X, W1) + b1    # Linear combination
    a1 = sigmoid(z1)           # Activation

    # Hidden to Output
    z2 = np.dot(a1, W2) + b2   # Linear combination
    y_pred = z2                # Since it's regression, no activation

    # ----- Compute Loss -----
    # Mean Squared Error
    loss = np.mean((y_pred - y) ** 2)

    # ----- Backward Propagation -----
    # Output layer error
    d_loss_y_pred = 2 * (y_pred - y) / y.size  # Derivative of MSE wrt y_pred

    # Gradients for W2 and b2
    d_z2 = d_loss_y_pred                      # Since y_pred = z2
    d_W2 = np.dot(a1.T, d_z2)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    # Hidden layer error
    d_a1 = np.dot(d_z2, W2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)

    # Gradients for W1 and b1
    d_W1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # ----- Update Weights and Biases -----
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

    # ----- Print Loss Every 100 Epochs -----
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

# ----- Testing the Neural Network -----
# Let's test the network with a new input, e.g., [9,10] -> should predict 11
# Since our training data only goes up to 10, we'll extend the series
test_input = np.array([[9, 10]]) / float(np.max(series))  # Normalize

# Forward pass
z1_test = np.dot(test_input, W1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, W2) + b2
y_test_pred = z2_test * float(np.max(series))  # Denormalize

print("\nTesting the Neural Network:")
print(f"Input: [9, 10]")
print(f"Predicted Output: {y_test_pred.flatten()[0]:.2f} (Expected: 11)")
