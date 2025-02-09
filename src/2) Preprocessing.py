from paddleocr import PaddleOCR
import cv2
import os
import pandas as pd
import gc
from tqdm import tqdm

def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_image = cv2.medianBlur(thresh, 3)
        
        del gray, enhanced_image, thresh
        return processed_image
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def initialize_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        use_gpu=True,
        max_threads=12
    )

def get_processed_files(output_file):
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            return set(existing_df['Image Name'].tolist()), existing_df
        except:
            print("Corrupted output file detected, starting fresh...")
            return set(), pd.DataFrame(columns=['Image Name', 'Extracted Text'])
    return set(), pd.DataFrame(columns=['Image Name', 'Extracted Text'])

def process_images_safely(image_folder, output_file, checkpoint_interval=50):
    ocr = initialize_ocr()
    processed_files, df = get_processed_files(output_file)
    
    image_files = sorted([
        f for f in os.listdir(image_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))
    ])

    # Filter unprocessed files
    unprocessed_files = [f for f in image_files if f not in processed_files]
    
    if not unprocessed_files:
        print("All images already processed!")
        return

    print(f"Resuming from {len(processed_files)} processed images, {len(unprocessed_files)} remaining")

    try:
        progress_bar = tqdm(unprocessed_files, desc="Processing Images")
        for idx, filename in enumerate(progress_bar):
            image_path = os.path.join(image_folder, filename)
            
            processed_image = preprocess_image(image_path)
            if processed_image is None:
                df = pd.concat([df, pd.DataFrame([[filename, "LOAD_ERROR"]], 
                               columns=['Image Name', 'Extracted Text'])])
                continue
            result = None
            try:
                result = ocr.ocr(processed_image, cls=True)
                text = ' '.join([line[1][0] for line in result[0]]) if result else ''
            except Exception as ocr_error:
                text = f"OCR_ERROR: {str(ocr_error)}"
                result = None   
            
            # Add new entry
            new_row = pd.DataFrame([[filename, text]], 
                                  columns=['Image Name', 'Extracted Text'])
            df = pd.concat([df, new_row], ignore_index=True)

            # Cleanup and checkpoint
            del processed_image, result
            gc.collect()

            if (idx + 1) % checkpoint_interval == 0:
                df.to_csv(output_file, index=False)
                progress_bar.set_postfix_str(f"Checkpoint saved at {idx+1} images")

    except KeyboardInterrupt:
        print("\nUser interruption detected. Saving progress...")
    except Exception as e:
        print(f"\nCritical error occurred: {str(e)}. Saving progress...")
    finally:
        df.to_csv(output_file, index=False)
        print(f"Final progress saved to {output_file}")

# Configuration
image_folder = r"D:/Amazon ML Challenge 2024/Dataset/archive/images/train"  # Change to the actual folder name where the images are stored
output_file = r"D:/Amazon ML Challenge 2024/outputs/output_paddleocr1.csv"  # Save the results to the working directory

# Start processing
process_images_safely(image_folder, output_file)