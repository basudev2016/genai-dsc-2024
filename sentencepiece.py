import collections
import math
import re

# Larger corpus with more varied sentences
corpus = [
    "Running fast is enjoyable.", "He runs faster than me.", "The runner's pace was consistent.",
    "Conversations are meaningful.", "Conductor is controlling the train.", 
    "Discover the universe and its wonders.", "Innovation drives success in business.",
    "Creating a strong foundation is essential for learning."
]

# Step 1: Initialize Vocabulary with Characters and Special Tokens
vocab = ['<start>', '<end>', '[UNK]', ' ']
char_set = sorted(set(''.join(corpus)))
vocab += char_set  # Start with individual characters
print(f"Initial Vocabulary (Characters): {vocab}\n")

# Count the frequency of each word in the corpus
word_freq = collections.Counter(corpus)
print(f"Word Frequencies: {word_freq}\n")

# Helper function to calculate unigram probabilities
def calculate_unigram_probs(corpus_tokens):
    token_counts = collections.Counter(token for sentence in corpus_tokens for token in sentence)
    total_count = sum(token_counts.values())
    return {token: math.log(count / total_count) for token, count in token_counts.items()}

# Tokenize the corpus into characters initially
corpus_tokens = [[char for char in sentence] for sentence in corpus]
print(f"Initial Tokenized Corpus (Characters): {corpus_tokens}\n")

# Helper function to merge subwords
def merge_vocab(pair, corpus_tokens):
    pattern = re.escape(pair[0] + ' ' + pair[1])
    new_corpus = [re.sub(pattern, pair[0] + pair[1], ' '.join(sentence)).split() for sentence in corpus_tokens]
    return new_corpus

# Step 2: Perform SentencePiece-like tokenization over multiple iterations
num_iterations = 20
for i in range(num_iterations):
    unigram_probs = calculate_unigram_probs(corpus_tokens)
    
    # Get subword pairs to merge
    pair_freqs = collections.Counter()
    for sentence in corpus_tokens:
        for j in range(len(sentence) - 1):
            pair_freqs[(sentence[j], sentence[j+1])] += 1

    # Find the best pair to merge based on frequency
    if not pair_freqs:
        break
    best_pair = max(pair_freqs, key=pair_freqs.get)

    # Add the best pair to the vocabulary
    merged_token = ''.join(best_pair)
    if merged_token not in vocab:
        vocab.append(merged_token)
    
    # Merge the best pair in the corpus
    corpus_tokens = merge_vocab(best_pair, corpus_tokens)
    
    print(f"Iteration {i + 1} - Added Subword: {merged_token}")
    print(f"Tokenized Corpus after Iteration: {corpus_tokens}\n")

# Step 3: Final vocabulary after SentencePiece-like tokenization
print(f"Final Subword Vocabulary: {vocab}")
