import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Generate a simple sequence dataset
def generate_sequence(length=50):
    return np.sin(np.linspace(0, 2 * np.pi, length))

# Prepare the dataset for the autoregressive model
def create_dataset(sequence, time_steps=10):
    X, y = [], []
    for i in range(len(sequence) - time_steps):
        X.append(sequence[i:i + time_steps])
        y.append(sequence[i + time_steps])
    return np.array(X), np.array(y)

# Generate a sequence
sequence = generate_sequence(1000)
time_steps = 10
X_train, y_train = create_dataset(sequence, time_steps)

# Reshape input to be [samples, time steps, features]
X_train = np.expand_dims(X_train, axis=-1)

# Build the autoregressive model
model = tf.keras.Sequential([
    layers.Input(shape=(time_steps, 1)),
    layers.LSTM(50, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Function to generate sequences using the trained model
def generate_sequence_from_model(model, start_sequence, length=50):
    generated = list(start_sequence)
    for _ in range(length - len(start_sequence)):
        input_seq = np.array(generated[-time_steps:])
        input_seq = input_seq.reshape((1, time_steps, 1))
        next_value = model.predict(input_seq, verbose=0)
        generated.append(next_value[0, 0])
    return np.array(generated)

# Function to plot and save the generated sequence
def plot_and_save(epoch, original_sequence, generated_sequence):
    plt.figure(figsize=(10, 5))
    
    # Plot original sequence
    plt.plot(original_sequence, color='blue', label='Original Sequence')
    
    # Plot generated sequence
    plt.plot(np.arange(len(original_sequence), len(original_sequence) + len(generated_sequence)),
             generated_sequence, color='red', label='Generated Sequence')
    
    plt.title(f"Generated Sequence at Epoch {epoch}")
    plt.legend()
    plt.savefig(f"autoregressive_generated_epoch_{epoch}.png")
    plt.close()

# Training loop with saving images every 100 epochs
epochs = 500
for epoch in range(1, epochs + 1):
    model.fit(X_train, y_train, epochs=1, verbose=0)
    
    if epoch % 100 == 0 or epoch == epochs:
        start_sequence = sequence[:time_steps]
        generated_sequence = generate_sequence_from_model(model, start_sequence, length=100)
        plot_and_save(epoch, sequence, generated_sequence)

# Generate the final sequence and save
start_sequence = sequence[:time_steps]
generated_sequence = generate_sequence_from_model(model, start_sequence, length=100)
plot_and_save(epochs, sequence, generated_sequence)
