"""
Prediction on Test Data and Post-Processing
This script loads a test CSV (with columns: index, image_link, group_id, entity_name),
creates a 'training_input' column if missing, uses the fine-tuned model to generate predictions,
and saves the output to a CSV file.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
TEST_DATA_PATH = r"E:/Hackethon/Amazon ML Challenge 24/archive/student_resource_3/dataset/test.csv"   
SAVE_DIRECTORY = r"E:/Hackethon/Amazon ML Challenge 24/model"
TEST_OUTPUT_PATH = r"E:/Hackethon/Amazon ML Challenge 24/outputs/final_test_dataset_output.csv" 
INFERENCE_BATCH_SIZE = 32

# --- LOAD TEST DATA ---
df_test = pd.read_csv(TEST_DATA_PATH)  # Expected columns: index, image_link, group_id, entity_name

# If 'training_input' column does not exist, create it from 'entity_name'
if 'training_input' not in df_test.columns:
    df_test['training_input'] = df_test['entity_name']

test_texts = df_test['training_input']

# --- LOAD THE SAVED MODEL & TOKENIZER ---
tokenizer = BartTokenizer.from_pretrained(SAVE_DIRECTORY)
model = BartForConditionalGeneration.from_pretrained(SAVE_DIRECTORY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Ensure the decoder_start_token_id is set
if model.config.decoder_start_token_id is None:
    if model.config.bos_token_id is not None:
        model.config.decoder_start_token_id = model.config.bos_token_id
    else:
        model.config.decoder_start_token_id = 2  # A common default for bart-base

# --- TOKENIZE THE TEST INPUTS ---
input_tokens = []
for text in test_texts:
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)['input_ids']
    input_tokens.append(tokens.squeeze(0).to(device))

# --- GENERATE PREDICTIONS ---
generated_outputs = []
for i in tqdm(range(0, len(input_tokens), INFERENCE_BATCH_SIZE), desc="Generating Batches"):
    inputs_batch = pad_sequence(input_tokens[i:i+INFERENCE_BATCH_SIZE], batch_first=True, padding_value=tokenizer.pad_token_id)
    with autocast():
        generated_ids = model.generate(
            inputs_batch,
            generation_config=GenerationConfig.from_pretrained(SAVE_DIRECTORY),
            max_length=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    for gen_ids in generated_ids:
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        generated_outputs.append(generated_text)

# --- SAVE THE OUTPUT ---
df_test['entity_value'] = generated_outputs
df_test.to_csv(TEST_OUTPUT_PATH, index=False)
print(f"Test predictions saved to: {TEST_OUTPUT_PATH}")