import cv2
import os
import numpy as np

def copy_and_paste(img_path, output_dir, paste_size=(64, 64), num_augmentations=3):
    # Read the image
    img = cv2.imread(img_path)

    for i in range(num_augmentations):
        # Randomly select a source region
        h, w, _ = img.shape
        y1 = np.random.randint(0, h - paste_size[0])
        x1 = np.random.randint(0, w - paste_size[1])
        source_region = img[y1:y1 + paste_size[0], x1:x1 + paste_size[1]]

        # Randomly select a target position
        y2 = np.random.randint(0, h - paste_size[0])
        x2 = np.random.randint(0, w - paste_size[1])

        # Copy and paste the source region to the target position
        img[y2:y2 + paste_size[0], x2:x2 + paste_size[1]] = source_region

    # Save to the specified path
    filename = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{filename}_aug.png"), img)

def batch_augment(img_dir, output_dir):
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        copy_and_paste(img_path, output_dir)

# Batch process all images in the directory containing HR2 images
batch_augment("E:/Python_study/breast/HR2/", "E:/Python_study/breast/HR2_aug/")
