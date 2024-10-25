from openai import OpenAI
import os

# Initialize the OpenAI client (API key fetched from environment automatically)
client = OpenAI()

# Zero-shot text classification example
def zero_shot_classification(prompt, categories):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Classify the following text into one of the categories: {categories}.\n\nText: {prompt}"}
        ],
        max_tokens=100,
        temperature=0.0,  # Set temperature low for deterministic output
    )
    return response.choices[0].message.content.strip()

# Example text to classify
text_to_classify = "The stock market crashed due to economic instability."

# Categories that the model hasn't been explicitly trained on
categories = "Entertainment, Politics, Sports, Technology, Business"

# Perform zero-shot classification
classification = zero_shot_classification(text_to_classify, categories)
print(f"Text: {text_to_classify}\nClassification: {classification}")
