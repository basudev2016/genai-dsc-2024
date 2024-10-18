from langchain.llms import OpenAI  # Correct import from Langchain
from langchain.prompts import PromptTemplate

# Read the prompt from the 'website_text.txt' file
with open('website_text.txt', 'r') as file:
    prompt = file.read()

# Define the hotel assistant template
hotel_assistant_template = prompt + """
You are the hotel manager of Landon Hotel, named "Mr. Landon". 
Your expertise is exclusively in providing information and advice about anything related to Landon Hotel. 
This includes any general Landon Hotel related queries. 
You do not provide information outside of this scope. 
If a question is not about Landon Hotel, respond with, "I can't assist you with that, sorry!" 
Question: {question} 
Answer: 
"""

# Create the prompt template
hotel_assistant_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=hotel_assistant_template
)

# Initialize the LLM (using Langchain's OpenAI integration)
llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0)

# Use the new chaining method with the pipe (`|`) operator
llm_chain = hotel_assistant_prompt_template | llm

# Function to query the LLM
def query_llm(question):
    response = llm_chain.invoke({'question': question})  # Use invoke method
    print(response)

# Interactive loop for querying the LLM
while True:
    user_input = input("Ask a question: ")
    query_llm(user_input)
