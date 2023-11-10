# Manually specify paths for an HR image and an LR image for model validation, defaults are 'HR/val/tile_3918.png' and 'LR_x4/val/tile_3918.png'
# This script generates a comparison of four images: "Original HR"; "Original LR"; "LR after Bilinear Interpolation"; "LR after SRCNN"
# Additionally, it calculates PSNR and SSIM values for images processed by "Bilinear Interpolation" and the "SRCNN model" respectively
# If the PSNR and SSIM values for the "SRCNN model" are higher than those of "Bilinear Interpolation", the model is considered effective

# PSNR and SSIM are two widely used metrics for evaluating image quality.
# PSNR (Peak Signal-to-Noise Ratio): PSNR calculates the mean squared error (MSE) between the original and reconstructed/compressed images and converts it to a logarithmic scale. Higher PSNR values indicate less difference between the reconstructed and original images, representing better image quality.
# SSIM (Structural Similarity Index): Measures the structural similarity between two images. It considers differences in brightness, contrast, and structure, calculating a similarity score between 0 and 1. Higher SSIM values indicate higher structural similarity and better image quality.

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

from EDSR_plus import EDSR

# Check if CUDA device is available, if yes, use CUDA, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model and move it to the specified device
model = EDSR().to(device)
# Load model weights and ensure the model weights are on the same device
model.load_state_dict(torch.load('PLUS_model_weights.pth', map_location=device))
# Set the model to evaluation mode
model.eval()

# Define image transformation: convert PIL image to tensor
transform = ToTensor()

# Define a function to compute PSNR and SSIM metrics for images
def compute_metrics(true_image, pred_image):
    true_image = to_pil_image(true_image)
    pred_image = to_pil_image(pred_image)
    true_image = np.array(true_image)
    pred_image = np.array(pred_image)

    psnr_value = psnr(true_image, pred_image, data_range=255)
    ssim_value, _ = ssim(true_image, pred_image, full=True, data_range=255, multichannel=True)

    return psnr_value, ssim_value

# Use matplotlib for image visualization
import matplotlib.pyplot as plt

def show_images(hr, lr, bicubic, srcnn_output):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    images = [hr, lr, bicubic, srcnn_output]
    titles = ['HR Image', 'LR Image', 'Bicubic Interpolation', 'SRCNN Output']

    for ax, img, title in zip(axes, images, titles):
        img = img.clamp(0, 1)
        ax.imshow(img.detach().permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Load images and convert them to tensors, ensuring they are on the specified device
hr_image_path = "./HR/val/tile_3135.png"
hr_image = Image.open(hr_image_path)
hr_tensor = transform(hr_image).unsqueeze(0).to(device)

lr_image_path = "./LR_x4/val/tile_3135.png"
lr_image = Image.open(lr_image_path)
lr_tensor = transform(lr_image).unsqueeze(0).to(device)

# Upsample the LR image using bicubic interpolation
bicubic_upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
bicubic_output = bicubic_upsample(lr_tensor)

# Make predictions using the model
srcnn_output = model(lr_tensor)

# Move the predicted results and original images to CPU for display
hr_img = hr_tensor.squeeze().cpu()
lr_img = lr_tensor.squeeze().cpu()
bicubic_img = bicubic_output.squeeze().cpu()
srcnn_img = srcnn_output.squeeze().cpu()

# Display images
show_images(hr_img, lr_img, bicubic_img, srcnn_img)

# Compute and output PSNR and SSIM metrics
bicubic_psnr, bicubic_ssim = compute_metrics(hr_img, bicubic_img)
srcnn_psnr, srcnn_ssim = compute_metrics(hr_img, srcnn_img)

print(f"Bicubic Interpolation: PSNR = {bicubic_psnr:.2f} dB, SSIM = {bicubic_ssim:.2f}")
print(f"SRCNN Output: PSNR = {srcnn_psnr:.2f} dB, SSIM = {srcnn_ssim:.2f}")
