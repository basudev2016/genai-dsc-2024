import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import os

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Generate a simple 2D dataset (e.g., a Gaussian distribution)
num_samples = 1000
x_train = np.random.randn(num_samples, 2)

# Define the normalizing flow layers (planar flow)
class PlanarFlow(layers.Layer):
    def __init__(self):
        super(PlanarFlow, self).__init__()
        self.w = self.add_weight(shape=(2, 1), initializer="random_normal", trainable=True)
        self.u = self.add_weight(shape=(2, 1), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(1,), initializer="random_normal", trainable=True)

    def call(self, x):
        linear_term = tf.matmul(x, self.w) + self.b
        # Applying the tanh activation and ensuring the output has the correct shape
        h = tf.tanh(linear_term)
        # Reshape u to ensure proper broadcasting
        u_hat = tf.matmul(self.u, tf.ones((1, tf.shape(x)[0])))
        return x + tf.transpose(u_hat) * h

# Stack multiple flows together
def build_normalizing_flow(num_flows=5):
    inputs = layers.Input(shape=(2,))
    x = inputs
    for _ in range(num_flows):
        x = PlanarFlow()(x)
    model = tf.keras.Model(inputs, x)
    return model

# Function to plot and save original and generated data
def plot_and_save(epoch, generated_samples):
    plt.figure(figsize=(10, 5))

    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(x_train[:, 0], x_train[:, 1], s=5, color='blue', label='Original Data')
    plt.title("Original Data")
    plt.axis('equal')

    # Generated data
    plt.subplot(1, 2, 2)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=5, color='red', label='Generated Data')
    plt.title(f"Generated Data at Epoch {epoch}")
    plt.axis('equal')

    plt.savefig(f"normalizing_flow_generated_epoch_{epoch}.png")
    plt.close()

# Instantiate and compile the Normalizing Flow model
nf_model = build_normalizing_flow(num_flows=5)
nf_model.compile(optimizer='adam', loss='mse')

# Train the Normalizing Flow model and save plots every 100 epochs
for epoch in range(1, 501):
    nf_model.fit(x_train, x_train, epochs=1, batch_size=64, verbose=0)
    if epoch % 100 == 0:
        generated_samples = nf_model.predict(np.random.randn(num_samples, 2))
        plot_and_save(epoch, generated_samples)

# Generate the final plot at the 500th epoch
generated_samples = nf_model.predict(np.random.randn(num_samples, 2))
plot_and_save(500, generated_samples)
