from collections import defaultdict, Counter

# Step 1: Define a simple corpus with sentences
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat saw the dog",
    "the dog saw the cat",
    "the cat and the dog"
]

# Step 2: Tokenize each sentence by splitting characters with spaces
def tokenize_sentence(sentence):
    return list(sentence.replace(" ", "_")) + ['</w>']  # Add end-of-sentence token

tokenized_corpus = [tokenize_sentence(sentence) for sentence in corpus]

# Print the tokenized corpus
print("Step 2 - Initial Tokenized Corpus:")
for sentence in tokenized_corpus:
    print(" ".join(sentence))
print("\n")

# Step 3: Count the frequency of character pairs
def get_pair_frequencies(tokenized_corpus):
    pairs = defaultdict(int)
    for sentence in tokenized_corpus:
        for i in range(len(sentence) - 1):
            pair = (sentence[i], sentence[i + 1])
            pairs[pair] += 1
    return pairs

# Step 4: Merge the most frequent pair
def merge_most_frequent_pair(tokenized_corpus, pair_to_merge):
    new_tokenized_corpus = []
    bigram = ''.join(pair_to_merge)
    for sentence in tokenized_corpus:
        new_sentence = []
        i = 0
        while i < len(sentence):
            if i < len(sentence) - 1 and (sentence[i], sentence[i + 1]) == pair_to_merge:
                new_sentence.append(bigram)  # Merge the pair
                i += 2
            else:
                new_sentence.append(sentence[i])
                i += 1
        new_tokenized_corpus.append(new_sentence)
    return new_tokenized_corpus

# Step 5: Perform BPE for a specified number of merges
def byte_pair_encoding(tokenized_corpus, num_merges):
    for merge_iteration in range(num_merges):
        # Get pair frequencies
        pair_frequencies = get_pair_frequencies(tokenized_corpus)
        if not pair_frequencies:
            break
        
        # Find the most frequent pair
        most_frequent_pair = max(pair_frequencies, key=pair_frequencies.get)
        print(f"Step 4 - Merge Iteration {merge_iteration + 1}: Most Frequent Pair: {most_frequent_pair}")
        
        # Merge the most frequent pair
        tokenized_corpus = merge_most_frequent_pair(tokenized_corpus, most_frequent_pair)
        
        # Print the updated tokenized corpus
        print(f"Updated Tokenized Corpus After Merge {merge_iteration + 1}:")
        for sentence in tokenized_corpus:
            print(" ".join(sentence))
        print("\n")
    
    return tokenized_corpus

# Perform BPE with 10 merges and print the intermediate steps
final_corpus = byte_pair_encoding(tokenized_corpus, num_merges=10)
