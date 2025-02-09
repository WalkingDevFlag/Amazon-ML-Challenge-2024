"""
Evaluation with BLEU and Exact Match Accuracy
This script loads the fine-tuned model and tokenizer, splits a portion from the training data
as an evaluation set, generates predictions, and computes evaluation metrics.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --- CONFIGURATION ---
TRAINING_DATA_PATH = r'E:/Hackethon/Amazon ML Challenge 24/outputs/processed_training_input.csv'
SAVE_DIRECTORY = r"E:/Hackethon/Amazon ML Challenge 24/model"
BATCH_SIZE = 8

# --- LOAD DATA ---
df_train = pd.read_csv(TRAINING_DATA_PATH)
X = df_train['training_input']
y = df_train['entity_value']

# Create evaluation split (using the same random state to ensure consistency):
_, X_eval, _, y_eval = train_test_split(X, y, random_state=104, test_size=0.25, shuffle=True)

# --- LOAD THE SAVED MODEL & TOKENIZER ---
tokenizer = BartTokenizer.from_pretrained(SAVE_DIRECTORY)
model = BartForConditionalGeneration.from_pretrained(SAVE_DIRECTORY)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Ensure required token IDs are defined:
if model.config.decoder_start_token_id is None:
    if model.config.bos_token_id is not None:
        model.config.decoder_start_token_id = model.config.bos_token_id
    else:
        model.config.decoder_start_token_id = 2  # For Bart-base, default is often 2

if model.config.bos_token_id is None:
    model.config.bos_token_id = model.config.decoder_start_token_id

# --- TOKENIZE EVALUATION INPUTS ---
input_tokens_eval = [
    tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)['input_ids'].squeeze(0).to(device)
    for text in X_eval
]

# --- EVALUATION LOOP ---
predicted_values = []
actual_values = list(y_eval)
smooth_fn = SmoothingFunction().method4
total_bleu_score = 0
correct_predictions = 0

for i in tqdm(range(0, len(input_tokens_eval), BATCH_SIZE), desc="Evaluating"):
    inputs_batch = pad_sequence(input_tokens_eval[i:i+BATCH_SIZE], batch_first=True, padding_value=tokenizer.pad_token_id)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs_batch,
            max_length=20,
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    for idx, gen_ids in enumerate(generated_ids):
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        predicted_values.append(generated_text)
        actual_text = actual_values[i + idx]
        bleu_score = sentence_bleu([actual_text.split()], generated_text.split(), smoothing_function=smooth_fn)
        total_bleu_score += bleu_score
        if generated_text == actual_text:
            correct_predictions += 1

average_bleu_score = total_bleu_score / len(predicted_values)
accuracy = correct_predictions / len(predicted_values)

print("Example Predictions:")
for i in range(min(10, len(predicted_values))):
    print(f"Predicted: {predicted_values[i]}, Actual: {actual_values[i]}")
print(f"Exact Match Accuracy: {accuracy * 100:.2f}%")
print(f"Average BLEU Score: {average_bleu_score:.4f}")