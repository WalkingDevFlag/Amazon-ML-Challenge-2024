import pandas as pd
import requests
import os
from concurrent.futures import ThreadPoolExecutor

# Define the path to your CSV file and the folder where you want to save images
csv_file_path = r'D:/Amazon ML Challenge 2024/Dataset/test.csv'  
download_folder = r'D:/Amazon ML Challenge 2024/images/test' 

# Create the download folder if it doesn't exist
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

def download_image(image_url):
    """Download a single image."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()  
        image_name = os.path.basename(image_url)
        image_path = os.path.join(download_folder, image_name)
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        print(f'Downloaded: {image_name}')
    except Exception as e:
        print(f'Failed to download {image_url}: {e}')

def main():
    # Read the CSV file
    data = pd.read_csv(csv_file_path)
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(download_image, data['image_link'])

if __name__ == '__main__':
    main()
