import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Sample corpus (you can use a larger one for better embeddings)
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "I love playing with my dog",
    "foxes are wild animals",
    "dogs are loyal animals",
    "quick brown fox jumps high",
    "animals like fox and dog live in the wild",
]

# Preprocess the corpus: Tokenize the sentences into words
tokenized_corpus = [sentence.split() for sentence in corpus]

# Train the Word2Vec model
# size: dimensionality of the word embeddings
# window: context window size
# min_count: minimum frequency of words to be included in the model
# sg: 1 for skip-gram, 0 for CBOW (Continuous Bag of Words)
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=3, min_count=1, sg=1)

# Words in the vocabulary
words = list(word2vec_model.wv.index_to_key)
print(f"Vocabulary: {words}\n")

# Get the embedding vector for a specific word, e.g., 'dog'
dog_vector = word2vec_model.wv['loyal']
print(f"Embedding vector for 'loyal':\n{dog_vector}\n")

# Visualize word embeddings using PCA (reduce to 2D for plotting)
def plot_embeddings(model):
    X = model.wv[words]  # Get the word vectors
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(result[:, 0], result[:, 1])
    
    # Annotate the points with the corresponding words
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    
    plt.title("Word2Vec Embeddings (PCA Visualization)")
    plt.show()

# Plot the word embeddings
plot_embeddings(word2vec_model)
