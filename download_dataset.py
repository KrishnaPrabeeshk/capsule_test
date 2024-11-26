import requests
import zipfile
import os

# URL of the dataset file (download URL from Kaggle dataset page)
dataset_url = "https://www.kaggle.com/api/v1/datasets/download/tongpython/cat-and-dog"

# Define the destination path in your project directory
project_directory = "datasets"  # Update with your project path
zip_file_path = os.path.join(project_directory, "cat_and_dog_dataset.zip")
extracted_path = os.path.join(project_directory, "cat_and_dog_dataset")

# Create the project directory if it doesn't exist
if not os.path.exists(project_directory):
    os.makedirs(project_directory)

# Download the dataset
print("Downloading dataset...")
response = requests.get(dataset_url, stream=True)
with open(zip_file_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)
print(f"Dataset downloaded to: {zip_file_path}")

# Extract the dataset
print("Extracting dataset...")
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(extracted_path)
print(f"Dataset extracted to: {extracted_path}")

# Clean up by removing the zip file
os.remove(zip_file_path)
print("Cleaned up zip file.")
