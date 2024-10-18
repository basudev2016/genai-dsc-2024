# pip install tensorflow-hub
# pip install --upgrade tensorflow tensorflow-hub


import tensorflow as tf
import tensorflow_hub as hub

# Load pre-trained ELMo model from TensorFlow Hub
elmo = hub.load("https://tfhub.dev/google/elmo/3")

# Define a sample sentence for testing
sentences = ["I love machine learning.", \
             "ELMo embeddings capture syntax and semantics.",\
                "Deep learning ia awesome"]

# Compute ELMo embeddings
elmo_embeddings = elmo.signatures["default"](tf.constant(sentences))["elmo"]

# Output the shape of the embeddings
print(elmo_embeddings.shape)  # Example: (2, 6, 1024) -> 2 sentences, 6 tokens each, 1024-dimensional embeddings

# Tokenize sentences (manually since ELMo doesn't have a tokenizer)
tokens = [sentence.split() for sentence in sentences]

# Print the ELMo embeddings for each token
for i, sentence_tokens in enumerate(tokens):
    print(f"\nSentence {i+1}:")
    for j, token in enumerate(sentence_tokens):
        print(f"Token: {token} - Embedding: {elmo_embeddings[i][j][:5]}...")  # Displaying first 5 dimensions for brevity