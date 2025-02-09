"""
Unit Correction via Fuzzy Matching (Final Output)
This script post-processes the predicted output by:
1. Extracting numerical values and units from the text.
2. Applying fuzzy matching to correct unit spellings (with progress bars).
3. Saving the final corrected output to a CSV file.
"""

import pandas as pd
import re
from fuzzywuzzy import process
from tqdm import tqdm

# Enable pandas progress_apply
tqdm.pandas(desc="Fuzzy Matching Correction")

# --- CONFIGURATION ---
PREDICTIONS_PATH = r"E:/Hackethon/Amazon ML Challenge 24/outputs/final_test_dataset_output.csv" 
FINAL_OUTPUT_PATH = r"E:/Hackethon/Amazon ML Challenge 24/outputs/corrected_file.csv"        

# --- LOAD PREDICTIONS ---
df_pred = pd.read_csv(PREDICTIONS_PATH)
 
def extract_number_and_unit(value):
    """Extracts a number and its associated unit using regex."""
    match = re.search(r'(\d+\.?\d*)\s*([a-zA-Z]+)', str(value))
    if match:
        return f"{match.group(1)} {match.group(2)}"
    return None

# (Optional step: if desired, extract basic clean values)
df_pred['cleaned_entity_value'] = df_pred['entity_value'].apply(extract_number_and_unit)

# --- FUZZY MATCHING FOR UNIT CORRECTION ---
# Define allowed units mapping per entity type (example values)
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon',
                    'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}
allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

def correct_spelling(entity_value, allowed_units, threshold=80):
    """Uses fuzzy matching to correct unit spelling in a prediction."""
    if pd.isna(entity_value) or entity_value.strip() == "":
        return ""
    try:
        value, unit = entity_value.split(maxsplit=1)
    except ValueError:
        return entity_value
    corrected_unit, score = process.extractOne(unit, allowed_units)
    if score >= threshold:
        return f"{value} {corrected_unit}"
    return entity_value

def handle_missing_values(entity_value):
    return "" if pd.isna(entity_value) or entity_value.strip() == "" else entity_value

# Apply fuzzy matching correction with progress bars
df_pred['corrected_entity_value'] = df_pred['entity_value'].progress_apply(lambda x: correct_spelling(x, allowed_units))
df_pred['corrected_entity_value'] = df_pred['corrected_entity_value'].progress_apply(lambda x: handle_missing_values(x))

# Prepare final DataFrame with properly renamed columns if desired
final_df = df_pred[['group_id', 'corrected_entity_value']].rename(
    columns={'corrected_entity_value': 'prediction', 'group_id': 'index'}
)

# Save the final corrected predictions
df_pred.to_csv(FINAL_OUTPUT_PATH, index=True)
print(f"Corrected file saved to: {FINAL_OUTPUT_PATH}")

# Optionally, inspect a few final outputs
final_df = pd.read_csv(FINAL_OUTPUT_PATH)
print("Final output sample:")
print(final_df.head())