from openai import OpenAI
import os

# Initialize the OpenAI client (API key fetched from environment automatically)
client = OpenAI()

# Few-shot text classification function
def few_shot_classification(prompt, examples, categories):
    # Constructing a few-shot prompt
    example_prompts = ""
    for example in examples:
        example_prompts += f"Text: {example['text']}\nClassification: {example['label']}\n\n"
    
    # Full prompt with explicit instruction to classify as per the examples
    full_prompt = (
        example_prompts +
        f"Classify the following text into one of the categories: {categories}.\n\n"
        f"Only classify as Positive or Negative based on the explicit examples provided. Avoid assuming positivity if ambiguous.\n\n"
        f"Text: {prompt}"
    )

    # Call OpenAI API to perform classification
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies text based on provided examples only."},
            {"role": "user", "content": full_prompt}
        ],
        max_tokens=50,
        temperature=0.0,  # Low temperature for deterministic output
    )
    
    return response.choices[0].message.content.strip()

# Example text to classify
text_to_classify = "The meal had a distinct flavor."

# Categories that the model will classify into
categories = "Positive, Negative"

# Few-shot examples for Positive classification
few_shot_examples_positive = [
    {"text": "The meal was fantastic and bursting with flavor.", "label": "Positive"},
    {"text": "I loved every bite; the food was amazing.", "label": "Positive"},
    {"text": "The atmosphere was delightful, and the food was superb.", "label": "Positive"},
    {"text": "Everything was wonderful, and I enjoyed the food greatly.", "label": "Positive"},
    {"text": "The chef's skills were impressive, and every dish was delicious.", "label": "Positive"}
]

# Few-shot examples for Negative classification
few_shot_examples_negative = [
    {"text": "The meal was tasteless and disappointing.", "label": "Negative"},
    {"text": "I disliked the food; it was bland and unenjoyable.", "label": "Negative"},
    {"text": "The atmosphere was awful, and the food was unappetizing.", "label": "Negative"},
    {"text": "The experience was unpleasant, and the food was bad.", "label": "Negative"},
    {"text": "The flavors were off, and I did not enjoy the meal.", "label": "Negative"}
]

# First, classify with Positive sentiment examples
print("Classifying with Positive sentiment examples...")
classification_positive = few_shot_classification(text_to_classify, few_shot_examples_positive, categories)
print(f"Text: {text_to_classify}\nClassification: {classification_positive}\n")

# Then, classify with Negative sentiment examples
print("Classifying with Negative sentiment examples...")
classification_negative = few_shot_classification(text_to_classify, few_shot_examples_negative, categories)
print(f"Text: {text_to_classify}\nClassification: {classification_negative}")
