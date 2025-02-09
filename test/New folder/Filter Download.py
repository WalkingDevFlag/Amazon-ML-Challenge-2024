import pandas as pd
import os
import requests
from urllib.parse import urlparse

# Load the filtered CSV file
csv_file = 'filtered_output.csv'  # Replace with the path to your filtered CSV
df = pd.read_csv(csv_file)

# Path to save images
save_folder = r'E:\Random Python Scripts\Amazon ML Challenge\Dataset\731432'  # Replace with the path to the folder where you want to save the images

# Create the folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Function to download and save images
def download_image(image_url, image_name):
    try:
        # Send HTTP request to the image URL
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            # Extract the file extension from the URL
            file_extension = os.path.splitext(urlparse(image_url).path)[-1]
            if file_extension == '':
                file_extension = '.jpg'  # Default to .jpg if no extension found

            # Save the image with the provided name
            image_path = os.path.join(save_folder, f'{image_name}{file_extension}')
            with open(image_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image saved as: {image_path}")
        else:
            print(f"Failed to download {image_url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")

# Iterate through each row of the DataFrame
for index, row in df.iterrows():
    image_url = row['image_link']
    entity_name = row['entity_name']
    
    # Extract the base image name from the URL (without extension)
    image_name = os.path.splitext(os.path.basename(urlparse(image_url).path))[0]
    
    # Combine the image name and entity name
    full_image_name = f"{image_name}_{entity_name}"

    # Download and save the image
    download_image(image_url, full_image_name)
