#!/usr/bin/env python3
import os
import shutil
import sys
import traceback

DEST_DIR = os.path.join("Data", "Data_raw")
os.makedirs(DEST_DIR, exist_ok=True)

try:
    import kagglehub
    path = kagglehub.dataset_download("mengcius/cinic10")
    print("Path to dataset files:", path)

    if os.path.isdir(path):
        for item in os.listdir(path):
            source_item = os.path.join(path, item)
            destination_item = os.path.join(DEST_DIR, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
            else:
                shutil.copy2(source_item, destination_item)
        print("Data successfully copied to", DEST_DIR)
    else:
        print("Warning: The returned path is not a directory. Check the output of kagglehub.dataset_download().")

except Exception as e:
    print("An error occurred during the download and copy process:")
    traceback.print_exc()
    sys.exit(1)
