import os
import shutil
import random

def split_dataset(src_dir, dest_dirs, train_filenames, val_filenames, test_filenames):
    for filename in train_filenames:
        for dest_dir in dest_dirs:
            shutil.move(os.path.join(src_dir, filename), os.path.join(dest_dir, 'train', filename))

    for filename in val_filenames:
        for dest_dir in dest_dirs:
            shutil.move(os.path.join(src_dir, filename), os.path.join(dest_dir, 'val', filename))

    for filename in test_filenames:
        for dest_dir in dest_dirs:
            shutil.move(os.path.join(src_dir, filename), os.path.join(dest_dir, 'test', filename))

# Define source directory and destination directories
src_folder = 'E:/Python_study/SRCNN_Pytorch/HR2'
dest_folders = ['E:/Python_study/SRCNN_Pytorch/LR2_x4', 'E:/Python_study/SRCNN_Pytorch/HR2']

# Create train, validation, and test directories
for dest_dir in dest_folders:
    for sub_folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, sub_folder), exist_ok=True)

# Get all file names and shuffle
filenames = os.listdir(src_folder)
filenames = [f for f in filenames if os.path.isfile(os.path.join(src_folder, f))]
random.shuffle(filenames)

# Define split ratios
train_ratio, val_ratio = 0.8, 0.1
train_count = int(len(filenames) * train_ratio)
val_count = int(len(filenames) * val_ratio)

# Get train, validation, and test file names
train_filenames = filenames[:train_count]
val_filenames = filenames[train_count:train_count + val_count]
test_filenames = filenames[train_count + val_count:]

# Perform the same split in three folders
split_dataset(src_folder, dest_folders, train_filenames, val_filenames, test_filenames)
