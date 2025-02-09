import kagglehub

# Download latest version
path = kagglehub.dataset_download("suvroo/amazon-ml-challenge")

print("Path to dataset files:", path)