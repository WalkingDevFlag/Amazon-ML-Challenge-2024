"""
This script merges OCR results with metadata, cleans the extracted text,
and constructs a unified training input by concatenating the entity name and 
the cleaned Extracted Text (separated by " | "). The merged data is then saved 
to a new CSV file.
"""

import os
import pandas as pd

def cleaning(text):
    """
    Clean the text by removing unwanted characters and forcing ASCII.
    """
    if isinstance(text, str):
        text = text.replace('[', '').replace(']', '').replace("'", "")
        text = text.encode('ascii', errors='ignore').decode()
        text = text.replace('\n', '').replace('\t', '').replace('/r', '')
        text = text.replace('/', '').replace('(', '').replace(')', '')
        text = text.replace('?', '').replace('!', '')
        text = text.replace('@', '').replace('<', '').replace('>', '')
        return text
    return ''

def main():
    # File paths (adjust if needed)
    ocr_csv  = r'E:/Hackethon/Amazon ML Challenge 24/outputs/output_paddleocr2.csv'         
    meta_csv = r'E:/Hackethon/Amazon ML Challenge 24/outputs/train_subset_dataset.csv'      
    output_csv = r'E:/Hackethon/Amazon ML Challenge 24/outputs/processed_training_input.csv'
    
    # Verify that files exist
    if not os.path.isfile(ocr_csv):
        print(f"OCR file not found: {ocr_csv}")
        return
    if not os.path.isfile(meta_csv):
        print(f"Metadata file not found: {meta_csv}")
        return
    
    # Load the OCR results and metadata CSV files
    df_ocr = pd.read_csv(ocr_csv)
    df_meta = pd.read_csv(meta_csv)
    
    print("OCR DataFrame shape:", df_ocr.shape)
    print("Metadata DataFrame shape:", df_meta.shape)
    
    # Normalize the key columns:
    # In metadata, used key column is 'image_name'
    df_meta['image_name'] = df_meta['image_name'].astype(str).str.strip().str.lower()
    
    # In OCR data, remove the .jpg extension from the "Image Name" column,
    # then strip whitespace and lower-case everything. Create a new key column 'image_name'.
    df_ocr['image_name'] = df_ocr['Image Name'].astype(str).str.strip().str.lower().str.replace('.jpg', '', regex=False)
    
    # Diagnostic prints to inspect key values
    print("Sample keys from metadata:", df_meta['image_name'].unique()[:5])
    print("Sample keys from OCR:", df_ocr['image_name'].unique()[:5])
    
    # Merge on the common image identifier (both DataFrames now have 'image_name')
    merged_df = pd.merge(df_meta, df_ocr, on='image_name', how='inner')
    print("Merged DataFrame shape:", merged_df.shape)
    
    if merged_df.empty:
        print("Warning: The merged DataFrame is empty. Check if the key columns have matching values.")
    
    # Clean the OCR "Extracted Text" column
    merged_df['Extracted Text'] = merged_df['Extracted Text'].apply(cleaning)
    
    # Create the unified training input by concatenating 'entity_name' and cleaned 'Extracted Text'
    merged_df['training_input'] = merged_df['entity_name'].astype(str).str.cat(
        merged_df['Extracted Text'].astype(str), sep=' | '
    )
    
    # Save the processed DataFrame to a CSV file
    merged_df.to_csv(output_csv, index=False)
    print(f"Processed training input saved to: {output_csv}")

if __name__ == '__main__':
    main()