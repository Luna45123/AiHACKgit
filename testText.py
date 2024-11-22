from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium", "gpt2-large", etc., for different sizes
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define a text prompt
prompt_text = "How to cook beef in pan"

# Tokenize input and generate text
input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
output = model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92, top_k=50)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("This output: -->> "+generated_text)
