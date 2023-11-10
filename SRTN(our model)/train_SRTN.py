import torch
from PIL import Image
import os
from SRTN import SRTN

# ————————————————————————————————Load Dataset——————————————————————————————————

# Import the Dataset class from PyTorch, which is the base class for building custom datasets
from torch.utils.data import Dataset, DataLoader

# Define the SRDataset class, which inherits from the Dataset class and is used to load high-resolution and low-resolution image data
class SRDataset(Dataset):
    # Initialization function, defining the behavior of the object when created
    def __init__(self, hr_dir, lr_dir, transform=None):
        # hr_dir: Folder path for high-resolution images
        self.hr_dir = hr_dir
        # lr_dir: Folder path for low-resolution images
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

        # Return the low-resolution and high-resolution images
        return lr_image, hr_image

# The transforms module is a submodule of the torchvision library, containing various image processing and data augmentation transformations. Used here to convert images to tensors.
from torchvision import transforms

# Define the image processing pipeline: convert image data to tensors
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Use the SRDataset class to load the training set, specifying the paths for high-resolution and low-resolution images
dataset = SRDataset(hr_dir='HR/train', lr_dir='LR_x4/train', transform=transform)
# Use DataLoader to load the dataset, set batch size to 8, and shuffle the data
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Use the SRDataset class to load the test set, specifying the paths for high-resolution and low-resolution images
test_dataset = SRDataset(hr_dir='HR/test', lr_dir='LR_x4/test', transform=transform)
# Use DataLoader to load the test set, set batch size to 8, and do not shuffle the data
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# ————————————————————————————————Define SRTN Model——————————————————————————————————

import torch.nn as nn  # Import the neural network module from PyTorch

# Check if the current device supports CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the SRTN model and move it to the appropriate device (CPU or CUDA)
model = SRTN().to(device)

# Define the mean squared error loss function
criterion = nn.MSELoss()
# Use the Adam optimizer with a learning rate of 0.001 (Adam algorithm: momentum + adaptive gradient descent algorithm)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ————————————————————————————————Model Training and Saving——————————————————————————————————

num_epochs = 10  # Set the total number of training epochs

# Start the loop for each training epoch
for epoch in range(num_epochs):
    # For each batch in the training data
    for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
        # Move low-resolution and high-resolution image data to the device (CPU or GPU)
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        # Forward pass: input low-resolution images to the model and get the output
        outputs = model(lr_imgs)
        # Calculate the loss between the output and the true high-resolution images
        loss = criterion(outputs, hr_imgs)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear all optimized gradients
        loss.backward()  # Backward pass: calculate the loss gradient
        optimizer.step()  # Update model parameters

        # Print the loss information for each step
        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # ——————————————————————Evaluate the Effectiveness of this Training Epoch————————————————————————
    # After each training epoch, calculate the loss on the test dataset
    model.eval()  # Set the model to evaluation mode, disabling dropout and batchnorm layers
    test_losses = []  # List to store all test losses
    with torch.no_grad():  # Disable gradient computation, saving memory/computational speed in evaluation mode
        for lr_imgs, hr_imgs in test_dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)  # Move test data to the device (CPU or GPU)

            # Make predictions using the model
            outputs = model(lr_imgs)
            # Calculate the test loss
            test_loss = criterion(outputs, hr_imgs)
            test_losses.append(test_loss.item())  # Add the current loss to the list

    # Calculate the average test loss
    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

    model.train()  # Set the model back to training mode, continuing to the next training epoch

# Save the model parameters to a file after training
torch.save(model.state_dict(), 'SRTN_model_weights.pth')
