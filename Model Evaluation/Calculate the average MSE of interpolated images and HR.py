import os
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. Given SRDataset class and data loading code...
# Define the SRDataset class, inheriting from the Dataset class, used for loading high-resolution and low-resolution image data
class SRDataset(Dataset):
    # Initialization function, defining the behavior of the object when created
    def __init__(self, hr_dir, lr_dir, transform=None):
        # hr_dir: Folder path containing high-resolution images
        self.hr_dir = hr_dir
        # lr_dir: Folder path containing low-resolution images
        self.lr_dir = lr_dir
        # transform: Image processing function/transform
        self.transform = transform
        # Get all filenames in the hr_dir folder
        self.filenames = os.listdir(hr_dir)

    # Return the total length of the dataset
    def __len__(self):
        return len(self.filenames)

    # Define how to get a single item from the dataset, i.e., get the image at a given index
    def __getitem__(self, idx):
        # Load the high-resolution image at the corresponding index
        hr_image = Image.open(os.path.join(self.hr_dir, self.filenames[idx]))
        # Load the low-resolution image at the corresponding index
        lr_image = Image.open(os.path.join(self.lr_dir, self.filenames[idx]))

        # If transform is defined, apply the corresponding processing/transform to the images
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        # Return the low-resolution image and high-resolution image
        return lr_image, hr_image

# Define image processing pipeline: convert image data to a tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Use the SRDataset class to load the test set, specifying the paths for high-resolution and low-resolution images
test_dataset = SRDataset(hr_dir='HR/test', lr_dir='LR_x4/test', transform=transform)
# Use DataLoader to load the test set with a batch size of 50 and no shuffling
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)

# 2. Define the MSE computation function
criterion = nn.MSELoss()

# 3. Upsample LR images using bicubic interpolation
def bicubic_upsample(lr_imgs, scale_factor=4):
    upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True)
    return upsample(lr_imgs)

# 4. Compute MSE between the upsampled images and HR images
def compute_mse(dataloader):
    total_mse = 0.0
    total_samples = 0

    for lr_imgs, hr_imgs in dataloader:
        upsampled_imgs = bicubic_upsample(lr_imgs)
        mse = criterion(upsampled_imgs, hr_imgs)
        total_mse += mse.item() * len(lr_imgs)  # Consider batch size using len(lr_imgs)
        total_samples += len(lr_imgs)

    return total_mse / total_samples

# Output the average MSE
print("Average MSE for the test dataset:", compute_mse(test_dataloader))
