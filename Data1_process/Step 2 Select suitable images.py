from PIL import Image
import numpy as np
import os
import shutil
import random


def calculate_stddev(img_path):
    """Calculate the standard deviation of an image"""
    img = Image.open(img_path)
    img_array = np.asarray(img.convert("L"))  # Convert to grayscale and then to a numpy array
    return np.std(img_array)


def random_select_images(input_folder, threshold, max_images=100):
    """Randomly select images with rich content from a folder"""
    filenames = [os.path.join(input_folder, fname) for fname in os.listdir(input_folder) if fname.endswith('.png')]
    random.shuffle(filenames)  # Shuffle the order of images randomly

    selected_files = []
    for file in filenames:
        if calculate_stddev(file) > threshold and len(selected_files) < max_images:
            selected_files.append(file)

    return selected_files


def copy_images_to_new_folder(selected_files, output_folder):
    """Copy images to a new folder"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file in selected_files:
        shutil.copy(file, output_folder)


input_folder = 'HR_svs'  # Change to your input folder path
threshold = 30  # You can adjust this value to find the best filtering effect
output_folder = 'HR'  # Change to your output folder path

selected_files = random_select_images(input_folder, threshold)
copy_images_to_new_folder(selected_files, output_folder)

print(f"Selected images copied to {output_folder} folder")
