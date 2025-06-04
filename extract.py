import os
import zipfile

DATASET_PATH = "./breast_images.zip"
EXTRACTED_PATH = "./Breast Histopathology Images"

def extract_dataset(dataset_path,extracted_path):
    if not os.path.exists(extracted_path):
        with zipfile.ZipFile(dataset_path,"r") as zip_ref:
            zip_ref.extractall(extracted_path)
    print(f"Dataset extracted to {extracted_path}")
    
extract_dataset(DATASET_PATH,EXTRACTED_PATH)