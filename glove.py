# https://nlp.stanford.edu/projects/glove/

import numpy as np

# Load the GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index

# Path to the GloVe file (after extraction)
glove_file = 'glove.6B.50d.txt'  # Use the 50d version, or change accordingly

# Load the embeddings
embeddings_index = load_glove_embeddings(glove_file)

print(f"Total words in GloVe embeddings: {len(embeddings_index)}")

# Example usage: Find embedding for the word 'computer'
word = "computer"
if word in embeddings_index:
    print(f"Embedding for '{word}':\n{embeddings_index[word]}")
else:
    print(f"Word '{word}' not found in the GloVe embeddings")

# Find the cosine similarity between two words
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

word1 = "computer"
word2 = "laptop"

if word1 in embeddings_index and word2 in embeddings_index:
    vec1 = embeddings_index[word1]
    vec2 = embeddings_index[word2]
    similarity = cosine_similarity(vec1, vec2)
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")
else:
    print(f"One or both words ('{word1}', '{word2}') not found in GloVe embeddings.")
