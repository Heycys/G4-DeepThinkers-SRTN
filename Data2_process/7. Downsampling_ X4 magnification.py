from PIL import Image
import os


def downsample_image(image_path, scale_factor):
    """Downsample image"""
    with Image.open(image_path) as img:
        width, height = img.size
        new_width = width // scale_factor
        new_height = height // scale_factor
        return img.resize((new_width, new_height), Image.ANTIALIAS)


def process_images(input_folder, output_folders, scale_factors):
    """Process and save downsampled images"""
    if not os.path.exists(input_folder):
        print(f"Input folder {input_folder} does not exist!")
        return

    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)

            for folder, scale in zip(output_folders, scale_factors):
                downsampled_image = downsample_image(image_path, scale)
                save_path = os.path.join(folder, filename)
                downsampled_image.save(save_path)


input_folder = "E:/Python_study/SRCNN_Pytorch/HR2"  # Set the input image path
output_folders = ['E:/Python_study/SRCNN_Pytorch/LR2_x4', 'E:/Python_study/SRCNN_Pytorch/LR2_x8']
scale_factors = [4, 8]

process_images(input_folder, output_folders, scale_factors)
print("Downsampling completed!")
