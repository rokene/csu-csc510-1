#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# import matplotlib.pyplot as plt

# --- Dataset Preparation ---

# Define the series
series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Function to create input-output pairs
def create_dataset(series, window_size):
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

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

# --- Building the Neural Network with TensorFlow ---

# Define the model
model = Sequential()

# Hidden layer with ReLU activation
model.add(Dense(units=2, activation='relu', input_shape=(window_size,)))

# Output layer with linear activation (since it's a regression task)
model.add(Dense(units=1, activation='linear'))

# Display the model's architecture
model.summary()

# --- Compiling the Model ---

# Compile the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error')

# --- Training the Neural Network ---

# Train the model
history = model.fit(X, y, epochs=1000, verbose=0)

# Print loss every 100 epochs
for i in range(0, 1000, 100):
    print(f"Epoch {i+100}/1000, Loss: {history.history['loss'][i+99]:.6f}")

# --- Evaluating the Neural Network ---

# Define a new input for testing
test_input = np.array([[9, 10]]) / float(np.max(series))  # Normalize

# Make a prediction
predicted_output = model.predict(test_input)

# Denormalize the output
predicted_number = predicted_output[0][0] * float(np.max(series))

print("\nTesting the Neural Network:")
print(f"Input: [9, 10]")
print(f"Predicted Output: {predicted_number:.2f} (Expected: 11)")

# --- Visualizing Training Loss ---

# Plot training loss over epochs
# plt.figure(figsize=(10,6))
# plt.plot(history.history['loss'])
# plt.title('Training Loss over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.grid(True)
# plt.show()

# --- User Input Functionality ---

def get_user_input():
    while True:
        try:
            user_input = input("\nEnter a sequence of two numbers separated by a comma (e.g., 5,6), or type 'exit' to quit: ")
            if user_input.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break
            # Split the input and convert to float
            input_values = user_input.split(',')
            if len(input_values) != 2:
                raise ValueError("Please enter exactly two numbers separated by a comma.")
            input_sequence = [float(num.strip()) for num in input_values]
            # Normalize the input
            normalized_input = np.array([input_sequence]) / float(np.max(series))
            # Make a prediction
            prediction = model.predict(normalized_input)
            # Denormalize the output
            predicted_next = prediction[0][0] * float(np.max(series))
            print(f"Predicted next number in the series: {predicted_next:.2f}")
        except ValueError as ve:
            print(f"Input Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# Call the user input function
get_user_input()