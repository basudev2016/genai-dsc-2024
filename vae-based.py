import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten the images for the VAE
x_train = np.reshape(x_train, (len(x_train), 28 * 28))
x_test = np.reshape(x_test, (len(x_test), 28 * 28))

# Define the dimensions
latent_dim = 2  # Number of latent space dimensions
input_dim = x_train.shape[1]  # Input dimension (28*28=784)

# Encoder
inputs = tf.keras.Input(shape=(input_dim,))
h = layers.Dense(256, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_h = layers.Dense(256, activation='relu')
decoder_mean = layers.Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# Custom VAE model class to include the loss
class VAE(Model):
    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z])
        self.decoder = tf.keras.Model(z, x_decoded_mean)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Calculate the VAE loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(inputs, reconstructed), axis=-1
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
            )
        )
        self.add_loss(reconstruction_loss + kl_loss)

        return reconstructed

# Instantiate and compile the VAE model
vae = VAE()
vae.compile(optimizer='adam')

# Train the VAE
vae.fit(x_train, x_train, epochs=10, batch_size=64, validation_data=(x_test, x_test))

# Generate new images by sampling points from the latent space
def generate_and_plot_images(x_test, num_samples=5):
    # Randomly sample points from the latent space
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))

    # Use the decoder to generate images from these points
    generated_images = vae.decoder(random_latent_vectors)

    # Reshape the images back to 28x28 for display
    generated_images = np.reshape(generated_images, (num_samples, 28, 28))

    # Plot original and generated images in a single figure
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        # Plot original images
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        # Plot generated images
        plt.subplot(2, num_samples, i + 1 + num_samples)
        plt.imshow(generated_images[i], cmap='gray')
        plt.title("Generated")
        plt.axis('off')

    plt.savefig("original_and_generated_mnist_vae.png")
    plt.show()

# Display and save 5 original and 5 generated images in a single figure
generate_and_plot_images(x_test, num_samples=5)
