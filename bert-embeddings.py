# pip install transformers torch

import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample input text
sentences = ["I love machine learning.", "ELMo embeddings capture syntax and semantics.","Deep learning ia awesome"]

# Tokenize input text
inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

# Convert token IDs back to tokens (words)
tokens = [tokenizer.convert_ids_to_tokens(input_ids) for input_ids in inputs['input_ids']]

# Pass tokens through BERT model to get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Extract embeddings (the last hidden states of the model)
embeddings = outputs.last_hidden_state

# Display tokens and their corresponding embeddings
for i, sentence_tokens in enumerate(tokens):
    print(f"\nSentence {i+1}:")
    for token, embedding in zip(sentence_tokens, embeddings[i]):
        print(f"Token: {token} - Embedding: {embedding[:5]}...")  # Display first 5 dimensions of the embedding for brevity
