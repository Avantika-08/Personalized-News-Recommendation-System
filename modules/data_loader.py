import zipfile
import os
import pandas as pd

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path, sep="\t", low_memory=False)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def extract_zip(file_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            target_path = os.path.join(extract_to, file)
            if os.path.exists(target_path):
                os.remove(target_path)
            zip_ref.extract(file, extract_to)
    print(f"Extracted files: {zip_ref.namelist()}")
    return zip_ref.namelist()




