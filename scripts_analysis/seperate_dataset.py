import os
import shutil
import sklearn
from sklearn.model_selection import train_test_split

def split_and_save_dataset(data_dir, output_dir, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3, random_seed=42):
    """
    Splits CIF files into train, validation, and test sets and saves them into respective folders.

    Args:
        data_dir (str): Path to the folder containing all CIF files.
        output_dir (str): Path to the folder where train/valid/test folders will be created.
        train_ratio (float): Proportion of data to use for training.
        val_ratio (float): Proportion of data to use for validation.
        test_ratio (float): Proportion of data to use for testing.
        random_seed (int): Seed for reproducibility.
    """
    # Check that the ratios add up to 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # List all CIF files
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".cif")]
    if not all_files:
        raise ValueError("No CIF files found in the specified directory.")

    # Train/Validation/Test Split
    train_files, temp_files = train_test_split(all_files, test_size=(val_ratio + test_ratio), random_state=random_seed)
    val_files, test_files = train_test_split(temp_files, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=random_seed)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy files to respective directories
    for file in train_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(train_dir, file))
    for file in val_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(valid_dir, file))
    for file in test_files:
        shutil.copy(os.path.join(data_dir, file), os.path.join(test_dir, file))

    print(f"Dataset split completed!")
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")
    print(f"Test set: {len(test_files)} files")
    print(f"Saved in: {output_dir}")


def split_directories_by_name(source_dir, target_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split subdirectories into train, valid, and test sets based on name order and a given ratio.

    Args:
        source_dir (str): Path to the main directory containing subdirectories.
        target_dir (str): Path to the target parent directory where the train, valid, and test folders will be created.
        split_ratio (tuple): A tuple of three floats representing the train, valid, and test split ratios. Should sum to 1.0.
    """
    # Validate the split ratio
    if len(split_ratio) != 3 or not all(isinstance(x, (int, float)) for x in split_ratio) or sum(split_ratio) != 1.0:
        raise ValueError("split_ratio must be a tuple of three floats that sum to 1.0")

    # Get all subdirectories sorted by name
    subdirs = sorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

    total_count = len(subdirs)
    if total_count == 0:
        raise ValueError(f"No subdirectories found in {source_dir}")

    # Calculate split sizes
    train_size = int(total_count * split_ratio[0])
    valid_size = int(total_count * split_ratio[1])
    test_size = total_count - train_size - valid_size  # Remaining for test

    train_subdirs = subdirs[:train_size]
    valid_subdirs = subdirs[train_size:train_size + valid_size]
    test_subdirs = subdirs[train_size + valid_size:]

    # Create output directories
    train_dir = os.path.join(target_dir, "train")
    valid_dir = os.path.join(target_dir, "valid")
    test_dir = os.path.join(target_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move subdirectories to corresponding sets
    for subdir in train_subdirs:
        shutil.move(os.path.join(source_dir, subdir), os.path.join(train_dir, subdir))
    for subdir in valid_subdirs:
        shutil.move(os.path.join(source_dir, subdir), os.path.join(valid_dir, subdir))
    for subdir in test_subdirs:
        shutil.move(os.path.join(source_dir, subdir), os.path.join(test_dir, subdir))

    print(f"Split completed:")
    print(f"  Training: {len(train_subdirs)} subdirectories")
    print(f"  Validation: {len(valid_subdirs)} subdirectories")
    print(f"  Testing: {len(test_subdirs)} subdirectories")


# Example usage
# Example usage
source_directory = "/data/chuanrui/data/crystal/all"  # Replace with the directory containing subfolders
target_directory = "/data/chuanrui/data/crystal/graphene_3x3_cif_ham"   # Replace with the target parent directory
split_ratios = (0.8, 0.1, 0.1)               # Customize train, valid, test ratios
split_directories_by_name(source_directory, target_directory, split_ratios)



# name = "graphene3x3"
# # Example usage
# data_dir = f"/data/chuanrui/data/crystal/cif/{name}/all"  # Path to folder containing all CIF files
# output_dir = f"/data/chuanrui/data/crystal/cif/{name}/"  # Where train/valid/test folders will be created

# split_and_save_dataset(data_dir, output_dir, train_ratio=0.5, val_ratio=0.2, test_ratio=0.3)