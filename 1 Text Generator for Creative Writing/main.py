# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model to evaluation mode for inference
model.eval()

# Function to generate creative text
def generate_text(starting_sentence, max_length=200, temperature=0.7, top_k=50):
    """
    Generate creative text based on a starting sentence.
    
    Args:
        starting_sentence (str): The input sentence to start the generation.
        max_length (int): Maximum length of the generated sequence (in tokens).
        temperature (float): Controls creativity (higher = more random, lower = more focused).
        top_k (int): Limits sampling to the top k most likely tokens for diversity.
    
    Returns:
        str: The generated text.
    """
    # Tokenize the input sentence
    input_ids = tokenizer.encode(starting_sentence, return_tensors='pt')
    
    # Generate text with the model
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,  # Enable sampling for creative output
            num_return_sequences=1
        )
    
    # Decode the generated tokens into readable text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Approximate 50-100 words by trimming if necessary
    words = generated_text.split()
    if len(words) > 100:
        generated_text = ' '.join(words[:100])
    
    return generated_text

# Main loop for user interaction
print("Welcome to the Creative Writing Text Generator!")
print("Enter a starting sentence to generate a creative writing prompt (e.g., 'The forest was silent until…').")
print("Type 'quit' to exit.")

while True:
    # Get user input
    starting_sentence = input("\nEnter starting sentence: ")
    
    # Check for exit condition
    if starting_sentence.lower() == 'quit':
        print("Exiting the text generator. Goodbye!")
        break
    
    # Generate and display the text
    try:
        generated_text = generate_text(starting_sentence)
        print("\nGenerated Text:\n", generated_text)
        word_count = len(generated_text.split())
        print(f"(Word count: {word_count})")
    except Exception as e:
        print(f"An error occurred: {e}. Please try again.")