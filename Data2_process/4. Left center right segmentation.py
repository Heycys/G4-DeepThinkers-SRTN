import cv2
import os

def crop_image(img_path, output_dir):
    # Read the image
    img = cv2.imread(img_path)

    # Crop three 256x256 images
    img_top_left = img[0:256, 0:256]  # Top-left
    img_bottom_right = img[204:460, 444:700]  # Bottom-right
    img_center = img[102:358, 222:478]  # Center

    # Save to the specified paths
    filename = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{filename}_top_left.png"), img_top_left)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_bottom_right.png"), img_bottom_right)
    cv2.imwrite(os.path.join(output_dir, f"{filename}_center.png"), img_center)

def batch_crop(img_dir, output_dir):
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        crop_image(img_path, output_dir)

# Batch process all images in the directory containing 400X images
batch_crop("E:/breast/400X/", "E:/Python_study/SRCNN_Pytorch/HR2/")
