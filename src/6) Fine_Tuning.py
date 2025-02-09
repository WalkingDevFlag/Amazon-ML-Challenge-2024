"""
Fine-Tuning A Conditional Generation Model
This script loads the training data, fine-tunes a Bart model using the input (training_input)
and target (entity_value) texts, and then saves the trained model, tokenizer, and generation configuration.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
TRAINING_DATA_PATH = r'E:/Hackethon/Amazon ML Challenge 24/outputs/processed_training_input.csv'
SAVE_DIRECTORY = r"E:/Hackethon/Amazon ML Challenge 24/model"
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# --- LOAD TRAINING DATA ---
df_train = pd.read_csv(TRAINING_DATA_PATH)
X = df_train['training_input']
y = df_train['entity_value']

# Create a training/evaluation split for monitoring (evaluation split used only during training)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)

def tokenize_function(text):
    return tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# --- INITIALIZE MODEL & TOKENIZER ---
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# --- PREPARE TRAINING TENSORS ---
# Tokenize training inputs and targets; each sample returns a tensor.
input_tokens = [tokenize_function(x)['input_ids'].squeeze(0) for x in X_train]
target_tokens = [tokenize_function(text)['input_ids'].squeeze(0) for text in y_train]

# --- SET DEVICE & TRAINING MODE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
input_tokens = [t.to(device) for t in input_tokens]
target_tokens = [t.to(device) for t in target_tokens]

# --- TRAINING LOOP ---
print("Starting fine-tuning...\n")
for epoch in range(EPOCHS):
    total_loss = 0
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    for i in tqdm(range(0, len(input_tokens), BATCH_SIZE), desc="Training Progress", unit="batch"):
        inputs_batch = pad_sequence(input_tokens[i:i+BATCH_SIZE], batch_first=True, padding_value=tokenizer.pad_token_id)
        targets_batch = pad_sequence(target_tokens[i:i+BATCH_SIZE], batch_first=True, padding_value=tokenizer.pad_token_id)
        optimizer.zero_grad()
        outputs = model(input_ids=inputs_batch, labels=targets_batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(input_tokens)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}\n")

print("Training complete.\n")

# --- SAVE THE MODEL, TOKENIZER, AND GENERATION CONFIGURATION ---
model.save_pretrained(SAVE_DIRECTORY)
tokenizer.save_pretrained(SAVE_DIRECTORY)
gen_config = GenerationConfig(
    early_stopping=True,
    num_beams=4,
    no_repeat_ngram_size=3,
    forced_bos_token_id=0,
    forced_eos_token_id=2
)
gen_config.save_pretrained(SAVE_DIRECTORY)
print(f"Model and tokenizer saved to {SAVE_DIRECTORY}")