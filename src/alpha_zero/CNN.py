import torch
import torch.nn as nn

# Define board dimensions
batch_size = 32
grid_size = 9
num_values = 6  # Possible values: 0-5
num_players = 2  # Player 1, Player 2

class CNN(nn.Module):
    def __init__(self, num_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * grid_size * grid_size, 128)
        self.fc2 = nn.Linear(128, grid_size * grid_size * num_channels)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(x.size(0), -1, grid_size, grid_size)  # Reshape to (batch_size, channels, height, width)
        return x

# Instantiate the model
model = CNN(num_channels=8)
print(model)