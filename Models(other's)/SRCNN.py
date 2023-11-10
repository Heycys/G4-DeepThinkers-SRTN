# Manually adjusted hyperparameters: batch_size=32 for training, batch_size=32 for testing, num_epochs = 10
# Batch_size is the batch processing amount. A larger value speeds up training but may cause errors due to insufficient system memory.
# Num_epochs is the number of training epochs. One epoch means training on the entire dataset. Ensure enough epochs to reduce loss to a certain value.
# Note: At the end of training, ensure that the final test loss is less than 0.0026.
# This is because LR images undergo bicubic interpolation preprocessing during input, and the average loss relative to HR images is 0.0026. If the model does not reduce this loss to below 0.0026, it indicates that the model has no practical effect.

import torch
from PIL import Image
import os

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

        if hr_image.shape != (3, 256, 256) or lr_image.shape != (3, 64, 64):
            print(f"Unexpected shape at index {idx}, filename {self.filenames[idx]}: {hr_image.shape}, {lr_image.shape}")

        return lr_image, hr_image

from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset = SRDataset(hr_dir='HR/train', lr_dir='LR_x4/train', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

test_dataset = SRDataset(hr_dir='HR/test', lr_dir='LR_x4/test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the SRCNN Model

import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
        self.layer1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRCNN().to(device)

# Loss Function and Optimizer

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
torch.save(model.state_dict(), 'srcnn_model_weights.pth')
