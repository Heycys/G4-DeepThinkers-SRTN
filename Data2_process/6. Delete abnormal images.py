import cv2
import os
import numpy as np

def check_conditions(img):
    # Convert to grayscale to evaluate pixel brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Number of pixels close to white
    white_pixels = np.sum(gray > 200)  # 200 is the brightness threshold, adjust as needed
    # Number of pixels close to black
    black_pixels = np.sum(gray < 55)  # 55 is the darkness threshold, adjust as needed

    total_pixels = img.shape[0] * img.shape[1]

    # If pixels close to white or close to black occupy a large portion of the image (e.g., over 30%), return True
    if white_pixels / total_pixels > 0.30 or black_pixels / total_pixels > 0.30:
        return True
    return False

directory = "E:/breast/HR2+HR2_aug"  # Your image directory
target_images = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Supported file formats
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if check_conditions(img):
            target_images.append(filename)
            os.remove(img_path)  # Delete images in the HR folder that meet the conditions

print("Deleted images in HR:", target_images)

# # Also delete files with the same name in the LR_x4 and LR_x8 folders
# folders = ['LR_x4', 'LR_x8']
#
# for folder in folders:
#     for filename in target_images:
#         img_path = os.path.join(folder, filename)
#         if os.path.exists(img_path):
#             os.remove(img_path)
#             print(f"Deleted {filename} from {folder}")
