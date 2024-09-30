#!/usr/bin/env python3

import numpy as np

# --- Activation Functions ---

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return x * (1 - x)

# --- Dataset Preparation ---

def create_dataset(series, window_size):
    """
    Creates input-output pairs from a numerical series.

    Parameters:
    - series (np.array): The numerical series.
    - window_size (int): The number of previous numbers to use as input.

    Returns:
    - X (np.array): Input features.
    - y (np.array): Target values.
    """
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

# --- Neural Network Structure ---

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initializes the neural network with random weights and biases.

        Parameters:
        - input_size (int): Number of input neurons.
        - hidden_size (int): Number of hidden neurons.
        - output_size (int): Number of output neurons.
        - learning_rate (float): Learning rate for gradient descent.
        """
        self.learning_rate = learning_rate
        # Weight initialization with mean 0
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(1, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(1, output_size)
    
    def forward(self, X):
        """
        Performs forward propagation.

        Parameters:
        - X (np.array): Input data.

        Returns:
        - y_pred (np.array): Predicted output.
        """
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        y_pred = self.z2  # Linear activation for regression
        return y_pred
    
    def compute_loss(self, y_pred, y):
        """
        Computes Mean Squared Error loss.

        Parameters:
        - y_pred (np.array): Predicted output.
        - y (np.array): True output.

        Returns:
        - loss (float): Computed loss.
        """
        loss = np.mean((y_pred - y) ** 2)
        return loss
    
    def backward(self, X, y, y_pred):
        """
        Performs backward propagation and updates weights and biases.

        Parameters:
        - X (np.array): Input data.
        - y (np.array): True output.
        - y_pred (np.array): Predicted output.
        """
        m = y.shape[0]  # Number of samples

        # Compute the gradient of loss w.r.t y_pred
        d_loss_y_pred = 2 * (y_pred - y) / m  # Shape: (m, 1)
        
        # Gradients for W2 and b2
        d_z2 = d_loss_y_pred  # Since y_pred = z2
        d_W2 = np.dot(self.a1.T, d_z2)  # Shape: (hidden_size, output_size)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)  # Shape: (1, output_size)
        
        # Gradients for W1 and b1
        d_a1 = np.dot(d_z2, self.W2.T)  # Shape: (m, hidden_size)
        d_z1 = d_a1 * sigmoid_derivative(self.a1)  # Shape: (m, hidden_size)
        d_W1 = np.dot(X.T, d_z1)  # Shape: (input_size, hidden_size)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)  # Shape: (1, hidden_size)
        
        # Update weights and biases
        self.W2 -= self.learning_rate * d_W2
        self.b2 -= self.learning_rate * d_b2
        self.W1 -= self.learning_rate * d_W1
        self.b1 -= self.learning_rate * d_b1
    
    def train(self, X, y, epochs=1000, print_loss=True):
        """
        Trains the neural network.

        Parameters:
        - X (np.array): Input data.
        - y (np.array): True output.
        - epochs (int): Number of training iterations.
        - print_loss (bool): Whether to print loss during training.
        """
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            
            # Backward propagation
            self.backward(X, y, y_pred)
            
            # Print loss at intervals
            if print_loss and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, input_seq):
        """
        Predicts the next number in the series based on input sequence.

        Parameters:
        - input_seq (np.array): Input sequence of shape (1, window_size).

        Returns:
        - predicted_num (float): Predicted next number.
        """
        y_pred = self.forward(input_seq)
        predicted_num = y_pred[0][0]
        return predicted_num

# --- Main Execution ---

def main():
    # Define the series
    series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Create input-output pairs
    window_size = 2
    X, y = create_dataset(series, window_size)
    
    # Normalize the data
    X = X / float(np.max(series))
    y = y / float(np.max(series))
    
    print("Input (X):")
    print(X)
    print("\nOutput (y):")
    print(y)
    
    # Initialize the Neural Network
    input_size = window_size    # 2
    hidden_size = 2             # Number of neurons in hidden layer
    output_size = 1             # 1 (predicting the next number)
    learning_rate = 0.1
    epochs = 1000
    
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    
    # Train the Neural Network
    nn.train(X, y, epochs=epochs, print_loss=True)
    
    # Testing the Neural Network
    test_input = np.array([[9, 10]]) / float(np.max(series))  # Normalize
    predicted_output = nn.predict(test_input)
    predicted_number = predicted_output * float(np.max(series))  # Denormalize
    
    print("\nTesting the Neural Network:")
    print(f"Input: [9, 10]")
    print(f"Predicted Output: {predicted_number:.2f} (Expected: 11)")
    
    # --- User Input Functionality ---
    
    while True:
        user_input = input("\nEnter a sequence of two numbers separated by a comma (e.g., 5,6), or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            print("Exiting the program. Goodbye!")
            break
        try:
            input_values = user_input.split(',')
            if len(input_values) != 2:
                raise ValueError("Please enter exactly two numbers separated by a comma.")
            input_sequence = [float(num.strip()) for num in input_values]
            # Normalize based on the dataset's maximum value
            normalized_input = np.array([input_sequence]) / float(np.max(series))
            # Predict
            prediction = nn.predict(normalized_input)
            # Denormalize
            predicted_num = prediction * float(np.max(series))
            print(f"Predicted next number in the series: {predicted_num:.2f}")
        except ValueError as ve:
            print(f"Input Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
