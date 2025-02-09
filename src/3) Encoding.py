import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv(input_path, output_path, error_text):
    """Process CSV file to handle OCR errors and prepare for BERT encoding"""
    try:
        df = pd.read_csv(input_path)
        # Replace OCR error text with NaN in second column
        df.iloc[:, 1] = df.iloc[:, 1].replace(error_text, pd.NA)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed CSV saved to {output_path}")
        return df
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")
        raise

def generate_bert_embeddings(df, text_column='Extracted Text', batch_size=32):
    """Generate BERT embeddings for text data with GPU support"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        df[text_column] = df[text_column].fillna('').astype(str)
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
        model.eval()
        
        embeddings = []
        texts = df[text_column].tolist()

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), 
                        desc="Processing batches",
                        unit="batch"):
                batch = texts[i:i+batch_size]
                valid = [(idx, t) for idx, t in enumerate(batch) if t.strip()]
                
                if not valid:
                    embeddings.extend([None]*len(batch))
                    continue
                
                indices, processed = zip(*valid)
                inputs = tokenizer(processed, 
                                 return_tensors='pt',
                                 padding=True,
                                 truncation=True,
                                 max_length=512).to(device)
                
                outputs = model(**inputs)
                batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                
                # Map results to original positions
                results = [None] * len(batch)
                for i, idx in enumerate(indices):
                    results[idx] = batch_emb[i]
                embeddings.extend(results)
        
        return pd.Series(embeddings, index=df.index)
    
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise

def main():
    """Main processing pipeline"""
    # Configuration
    input_csv = r'D:/Amazon ML Challenge 2024/outputs/output_paddleocr1.csv'
    processed_csv = r'D:/Amazon ML Challenge 2024/outputs/output_paddleocr2.csv'
    final_output = r'D:/Amazon ML Challenge 2024/outputs/train_extracted_text_with_embeddings.csv'
    error_text = "OCR_ERROR: 'NoneType' object is not iterable"
    
    try:
        # Process CSV
        df = process_csv(input_csv, processed_csv, error_text)
        
        # Generate embeddings
        logging.info("Starting BERT embedding generation")
        df['BERT_Embedding'] = generate_bert_embeddings(df)
        
        # Save final output
        df.to_csv(final_output, index=False)
        logging.info(f"Final output saved to {final_output}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
