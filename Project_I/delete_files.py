import os
import random
import argparse
import logging


def delete_files_in_folder(folder_path: str, delete_percent: float) -> None:
    """
    Deletes a given percentage of files in the specified folder and all its subfolders.

    Parameters:
    - folder_path: Path to the folder containing the files.
    - delete_percent: The percentage of files to delete (between 0 and 1).
    """
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        logging.error(f"Folder {folder_path} does not exist.")
        return

    # Walk through all files in the directory and its subdirectories
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    # Calculate the number of files to delete
    num_files_to_delete = int(len(all_files) * delete_percent)

    if num_files_to_delete == 0:
        logging.warning("No files to delete (delete_percent too small).")
        return

    # Randomly sample files to delete
    files_to_delete = random.sample(all_files, num_files_to_delete)

    # Delete the selected files
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            logging.info(f"Deleted file: {file_path}")
        except Exception as e:
            logging.error(f"Failed to delete file {file_path}: {e}")


# === MAIN FUNCTION ===

def main():
    parser = argparse.ArgumentParser(description="Delete a given percentage of files in a folder and its subfolders.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the files to delete.")
    parser.add_argument("delete_percent", type=float, help="Percentage of files to delete (0 to 1).")

    args = parser.parse_args()

    # Validate delete_percent
    if args.delete_percent < 0 or args.delete_percent > 1:
        logging.error("delete_percent must be between 0 and 1.")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logging.info(f"Starting deletion process in folder: {args.folder_path}")
    delete_files_in_folder(args.folder_path, args.delete_percent)
    logging.info("Deletion process completed.")


if __name__ == "__main__":
    main()