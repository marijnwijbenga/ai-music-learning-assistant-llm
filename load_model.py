import torch
from utils.allowed_topics_validator import is_allowed_topic
from transformers import AutoModelForCausalLM, AutoTokenizer

# Select device to run on
device = torch.device("mps") if torch.has_mps else torch.device("cpu")

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad_token to eos_token if not available
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model
print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, revision="v0.1").to(device)
print(f"Model {model_name} loaded successfully.")

# Set pad_token_id to eos_token_id to handle padding properly
model.config.pad_token_id = model.config.eos_token_id

MAX_TOKENS = 100
TEMPERATURE = 0.2
TOP_P = 0.9

RESPONSE_LIMIT_TEXT = f"Please limit your response to {MAX_TOKENS} words, treat punctuation as a word."
ON_TOPIC_TEXT = "We are straying off-topic. Let us return to the topic of guitar studying."
messages = []
messages.append(
    { "role": "user",
     "content": f"{RESPONSE_LIMIT_TEXT}"})

def query_model(prompt, messages):
    # Add user prompt to messages
    messages.append({"role": "user", "content": prompt})

    # Extract just the content of all messages (user and assistant)
    topics = ' '.join([msg["content"] for msg in messages])
    
    print('topics as string: ', topics)
    # Commented validation of topic for now
    if not is_allowed_topic(topics):
        response = "We are straying off-topic. Let us return to the topic of guitar studying."
        return response, messages
    
    # Use the tokenizer's chat template to format input_text
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Tokenize the prompt and ensure padding is added if needed, also generate the attention mask
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    print('inputs', inputs)
    
    # Generate response from the model
    outputs = model.generate(
        inputs, 
        max_new_tokens=MAX_TOKENS, 
        temperature=TEMPERATURE, 
        top_p=TOP_P, 
        do_sample=True
    )
    
    # Decode the generated tokens to a string
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[:response.rfind('.')] + '.'  # trim last incomplete sentence
    
    # Add assistant response to messages
    messages.append({"role": "assistant", "content": response})
     
    return response, messages

# Run the session in a loop
while True:
    # Take user input
    prompt = input("You: ")
    
    # Exit condition
    if prompt.lower() == "exit":
        print("Session closed.")
        break
    
    # Get the model's response and update messages
    response, messages = query_model(prompt, messages)
    
    
    # Print AI's response
    print(f"AI: {response}")

