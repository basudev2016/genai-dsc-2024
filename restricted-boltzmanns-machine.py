import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import os

# Load the MNIST dataset
digits = load_digits()
data = digits.data / 16.0  # Scale the data to [0, 1]
n_samples, n_features = data.shape

# Define RBM parameters
n_hidden = 64  # Number of hidden units
n_visible = n_features  # Number of visible units (same as number of features)
epochs = 500
learning_rate = 0.1

# Initialize weights and biases
np.random.seed(42)
weights = np.random.normal(scale=0.1, size=(n_visible, n_hidden))
visible_bias = np.zeros(n_visible)
hidden_bias = np.zeros(n_hidden)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Ensure the output directory exists
output_dir = "rbm_outputs"
os.makedirs(output_dir, exist_ok=True)

# Training the RBM
print("Starting RBM training...")
for epoch in range(1, epochs + 1):
    # Positive phase
    pos_hidden_activations = np.dot(data, weights) + hidden_bias
    pos_hidden_probs = sigmoid(pos_hidden_activations)
    pos_hidden_states = pos_hidden_probs > np.random.rand(n_samples, n_hidden)
    pos_associations = np.dot(data.T, pos_hidden_probs)

    # Negative phase (reconstruct)
    neg_visible_activations = np.dot(pos_hidden_states, weights.T) + visible_bias
    neg_visible_probs = sigmoid(neg_visible_activations)
    neg_hidden_activations = np.dot(neg_visible_probs, weights) + hidden_bias
    neg_hidden_probs = sigmoid(neg_hidden_activations)

    # Update weights and biases
    neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
    weights += learning_rate * ((pos_associations - neg_associations) / n_samples)
    visible_bias += learning_rate * np.mean(data - neg_visible_probs, axis=0)
    hidden_bias += learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

    # Save the reconstructed images every 100 epochs and at the final epoch
    if epoch % 100 == 0 or epoch == epochs:
        print(f"Saving images at epoch {epoch}...")
        # Visualize original and reconstructed images
        fig, axs = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            # Original images
            axs[0, i].imshow(data[i].reshape(8, 8), cmap='gray')
            axs[0, i].axis('off')

            # Reconstructed images
            axs[1, i].imshow(neg_visible_probs[i].reshape(8, 8), cmap='gray')
            axs[1, i].axis('off')

        plt.suptitle(f"Original (top) vs Reconstructed (bottom) at Epoch {epoch}")
        output_path = os.path.join(output_dir, f"rbm_generated_vs_original_epoch_{epoch}.png")
        plt.savefig(output_path)
        plt.close()

print("Training complete!")
