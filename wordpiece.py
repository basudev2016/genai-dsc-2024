import collections
import re

# Larger corpus with more varied words
corpus = [
    "running fast", "runs faster", "runner's pace", "the fastest runner", "conversation", "converse",
    "construction", "conductor", "conditioning", "universal", "universe", "discover", "discovery",
    "multinational", "innovation", "creating", "creating a strong foundation", "foundational learning"
]

# Step 1: Initialize Vocabulary with Characters and Special Tokens
vocab = ['<start>', '<end>', '[UNK]'] + sorted(set(''.join(corpus)))
print(f"Initial Vocabulary (Characters): {vocab}\n")

# Count the frequency of each word in the corpus
word_freq = collections.Counter(corpus)
print(f"Word Frequencies: {word_freq}\n")

# Helper functions to tokenize and merge subwords
def get_stats(corpus_tokens):
    pairs = collections.defaultdict(int)
    for tokens in corpus_tokens:
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs

def merge_vocab(pair, corpus_tokens):
    pattern = ' '.join(pair)
    replacement = ''.join(pair)
    corpus_tokens = [re.sub(pattern, replacement, ' '.join(tokens)).split() for tokens in corpus_tokens]
    return corpus_tokens

# Tokenize the corpus initially into characters
corpus_tokens = [[char for char in word] for word in corpus]
print(f"Initial Tokenized Corpus (Characters): {corpus_tokens}\n")

# Step 2: Perform WordPiece tokenization over multiple iterations
num_iterations = 20
for i in range(num_iterations):
    pairs = get_stats(corpus_tokens)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    corpus_tokens = merge_vocab(best_pair, corpus_tokens)
    new_token = ''.join(best_pair)
    vocab.append(new_token)
    print(f"Iteration {i + 1} - Added Subword: {new_token}")
    print(f"Tokenized Corpus after Iteration: {corpus_tokens}\n")

# Step 3: Final vocabulary after WordPiece tokenization
print(f"Final Subword Vocabulary: {vocab}")
