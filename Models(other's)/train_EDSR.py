# Manually adjusted hyperparameters: batch_size=8 for training, batch_size=8 for testing, num_epochs = 10
# Batch_size is the batch processing amount. A larger value speeds up training but may cause errors due to insufficient system memory.
# Num_epochs is the number of training epochs. One epoch means training on the entire dataset. Ensure enough epochs to reduce loss to a certain value.
# Note: At the end of training, ensure that the final test loss is satisfactory.

import torch
from PIL import Image
import os
from EDSR import EDSR

# Load the Dataset

from torch.utils.data import Dataset, DataLoader

class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.transform = transform
        self.filenames = os.listdir(hr_dir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        hr_image = Image.open(os.path.join(self.hr_dir, self.filenames[idx]))
        lr_image = Image.open(os.path.join(self.lr_dir, self.filenames[idx]))

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = SRDataset(hr_dir='HR/train', lr_dir='LR_x4/train', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

test_dataset = SRDataset(hr_dir='HR/test', lr_dir='LR_x4/test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define the EDSR Model

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EDSR().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Saving the Model

num_epochs = 10

for epoch in range(num_epochs):
    for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        outputs = model(lr_imgs)
        loss = criterion(outputs, hr_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    model.eval()
    test_losses = []
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_dataloader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            outputs = model(lr_imgs)
            test_loss = criterion(outputs, hr_imgs)
            test_losses.append(test_loss.item())

    avg_test_loss = sum(test_losses) / len(test_losses)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}")

    model.train()

# Save the trained model parameters to a file
torch.save(model.state_dict(), 'EDSR_model_weights.pth')
