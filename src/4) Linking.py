# Summary:
# This script loads a CSV dataset and extracts image names from URL links.
# It constructs full file paths for images from a specified folder,
# filters out records where corresponding images are missing, and selects key columns.
# Finally, it saves the processed data (including the image_link and image_name columns) to a new CSV file,
# with progress bars tracking key steps.

import pandas as pd
import os
from tqdm.auto import tqdm

tqdm.pandas(desc="Processing")

# Load the dataset
train_subset = pd.read_csv(r'E:/Hackethon/Amazon ML Challenge 24/archive/student_resource_3/dataset/train.csv')

# Extract the image name from the URL with a progress bar
train_subset['image_name'] = train_subset['image_link'].progress_apply(lambda x: x.split('/')[-1].split('.')[0])

# Path to the unzipped images folder
image_folder_path = r'E:/Hackethon/Amazon ML Challenge 24/archive/archive/images/train'  

# Function to check if the image exists and return the file path
def get_image_path(image_name):
    image_file = f"{image_name}.jpg"
    image_path = os.path.join(image_folder_path, image_file)
    if os.path.exists(image_path):
        return image_path
    return None

# Add the image file paths to the dataframe using progress_apply
train_subset['image_file'] = train_subset['image_name'].progress_apply(get_image_path)

# Filter out rows where the image file is not found
df_filtered = train_subset.dropna(subset=['image_file'])

# Select the required columns including image_link and image_name
final_df = df_filtered[['image_link', 'image_name', 'image_file', 'group_id', 'entity_name', 'entity_value']]

# Save the final dataset to a new CSV file
final_df.to_csv(r'E:/Hackethon/Amazon ML Challenge 24/outputs/train_subset_dataset.csv', index=False)