import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample data: A set of sentences
corpus = [
       "Once upon a time there was a princess who loved to explore the forest.",
    "The forest was full of magical creatures and hidden treasures.",
    "One day, while wandering in the woods, she found a secret passage.",
    "The passage led to an ancient castle that was abandoned for centuries.",
    "Inside the castle, she discovered a library filled with old books.",
    "She spent hours reading about the history of the kingdom and its legends.",
    "Among the books, she found a map that pointed to a hidden treasure.",
    "Determined to find the treasure, she set out on a new adventure.",
    "Her journey was filled with challenges, but she never gave up.",
    "With courage and determination, she finally found the treasure.",
    "The treasure was not gold or jewels, but a magical artifact.",
    "The artifact had the power to bring peace and prosperity to the kingdom.",
    "She returned to the kingdom and used the artifact to help her people.",
    "The kingdom flourished, and the princess was celebrated as a hero.",
    "She continued to explore the world, seeking knowledge and adventure.",
    "Her stories inspired others to follow their dreams and be brave.",
    "In the end, the princess realized that the greatest treasure was the journey itself.",
    "She wrote down her adventures in a book that was passed down through generations.",
    "The book became a legend, inspiring countless others to explore the world.",
    "And so, the princess's legacy lived on, in the hearts of all who read her story.",
    "In a distant land, there was a small village surrounded by mountains.",
    "The villagers were known for their hospitality and kindness.",
    "One winter, a traveler arrived at the village, seeking shelter from the cold.",
    "The villagers welcomed the traveler and offered him food and warmth.",
    "As the traveler shared stories of his journeys, the villagers listened with awe.",
    "He spoke of faraway places, of deserts, forests, and cities.",
    "The villagers had never left their mountains, but they dreamed of the world beyond.",
    "The traveler stayed for a few days, and when he left, he gave the villagers a gift.",
    "The gift was a small, mysterious box that he told them never to open.",
]

# Tokenize the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]

# Convert labels to one-hot encoding
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Define the Energy-Based Model (EBM) for text generation
class TextEnergyModel(Model):
    def __init__(self, total_words, embed_dim, hidden_units):
        super(TextEnergyModel, self).__init__()
        self.embedding = layers.Embedding(total_words, embed_dim, input_length=max_sequence_len-1)
        self.lstm = layers.LSTM(hidden_units)
        self.dense = layers.Dense(total_words, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        energy = self.dense(x)
        return energy

# Instantiate the model
embed_dim = 64
hidden_units = 100
model = TextEnergyModel(total_words, embed_dim, hidden_units)

# Compile the model with categorical crossentropy as the loss function
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(xs, ys, epochs=500, verbose=1)

# Function to generate text using the model
def generate_text(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs, axis=1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word
    return seed_text

# Example usage: Generating text continuation
seed_text = "Once upon a time"
next_words = 10
print(generate_text(seed_text, next_words, max_sequence_len))
