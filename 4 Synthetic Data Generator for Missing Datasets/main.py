from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import re
import random

# --- Step 1: Load the Model and Tokenizer ---
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using M1 GPU (MPS) for acceleration.")
else:
    device = torch.device("cpu")
    print("MPS not available, falling back to CPU.")
model.to(device)

# --- Step 2: Define a Prompt ---
domain_prompt = """
Generate a customer review like this:
Name: [e.g., Alice Brown]
Rating: [e.g., 4]
Comment: [e.g., Great product, fast shipping]

Review:
Name: """

# --- Step 3: Function to Generate Synthetic Data ---
def generate_synthetic_review():
    """
    Generate a single synthetic customer review using DistilGPT2 or fallback logic.
    
    Returns:
        dict: Review with 'Name', 'Rating', and 'Comment'.
    """
    print("Starting generation...")
    inputs = tokenizer(domain_prompt, return_tensors="pt").to(device)
    print("Inputs prepared.")
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=150,
            num_beams=5,
            temperature=1.2,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated_text}\n")
    
    # Parse the generated text
    try:
        lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
        print(f"Lines after filtering:\n{lines}\n")
        
        review_start = -1
        for i, line in enumerate(lines):
            if "Review:" in line:
                review_start = i + 1
                break
        
        if review_start == -1 or len(lines) < review_start + 1:
            raise ValueError("No review section found.")
        
        name = rating = comment = None
        for line in lines[review_start:]:
            if re.match(r"Name: .+", line):
                name = re.search(r"Name: (.+)", line).group(1).strip()
            elif re.match(r"Rating: \d", line):
                rating = int(re.search(r"Rating: (\d)", line).group(1))
            elif re.match(r"Comment: .+", line):
                comment = re.search(r"Comment: (.+)", line).group(1).strip()
        
        if not all([name, rating, comment]) or name in ["[e.g., Alice Brown]", "______________"]:
            raise ValueError(f"Invalid review: Name={name}, Rating={rating}, Comment={comment}")
        
        rating = max(1, min(5, rating))
        print(f"Parsed review: Name={name}, Rating={rating}, Comment={comment}")
        return {"Name": name, "Rating": rating, "Comment": comment}
    except (AttributeError, IndexError, ValueError) as e:
        print(f"Error parsing review: {e}. Using varied fallback.")
        first_names = ["Emma", "Liam", "Olivia", "Noah", "Ava", "Ethan", "Sophia", "Mason"]
        last_names = ["Smith", "Johnson", "Brown", "Taylor", "Wilson", "Davis"]
        comments = [
            "Good product, shipping was okay this time.",
            "Really enjoyed this item, fast service!",
            "Not great, delivery took too long sadly.",
            "Fantastic quality, will definitely buy again.",
            "Average experience, nothing really stood out."
        ]
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        rating = random.randint(1, 5)
        comment = random.choice(comments)
        print(f"Fallback review: Name={name}, Rating={rating}, Comment={comment}")
        return {"Name": name, "Rating": rating, "Comment": comment}

# --- Step 4: Generate and Save Dataset ---
def create_synthetic_dataset(num_rows=10):
    """
    Generate a dataset of synthetic reviews and save to CSV.
    
    Args:
        num_rows (int): Number of rows to generate (default 10).
    """
    reviews = []
    for i in range(num_rows):
        print(f"\nGenerating review {i+1}/{num_rows}")
        review = generate_synthetic_review()
        reviews.append(review)
    
    df = pd.DataFrame(reviews, columns=["Name", "Rating", "Comment"])
    df.to_csv("synthetic_customer_reviews.csv", index=False)
    print(f"\nGenerated {len(df)} synthetic reviews. Saved to 'synthetic_customer_reviews.csv'.")
    
    print("\nDataset Statistics:")
    print(f"Mean Rating: {df['Rating'].mean():.2f}")
    print(f"Rating Variance: {df['Rating'].var():.2f}")
    print(f"Sample Review:\n{df.iloc[0]}")

# --- Step 5: Run the Generator ---
if __name__ == "__main__":
    print("Starting Synthetic Data Generation...")
    create_synthetic_dataset(num_rows=10)