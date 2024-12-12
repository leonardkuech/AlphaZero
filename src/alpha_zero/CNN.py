import torch
import torch.nn as nn

class BoardCNNWithPoints(nn.Module):
    def __init__(self, num_channels, grid_size, num_points_features):
        """
        Args:
            num_channels (int): Number of channels in the board input.
            grid_size (int): Width and height of the board (assumes square grid).
            num_points_features (int): Number of features in the player's points array.
        """
        super(BoardCNNWithPoints, self).__init__()
        self.grid_size = grid_size

        # CNN for processing the board state
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the flattened output size of the CNN
        self.cnn_output_size = 64 * grid_size * grid_size

        # Fully connected layers
        self.fc1 = nn.Linear(self.cnn_output_size + num_points_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Example for a regression or scalar output

    def forward(self, board, points):
        """
        Args:
            board (torch.Tensor): Tensor of shape (batch_size, num_channels, grid_size, grid_size).
            points (torch.Tensor): Tensor of shape (batch_size, num_points_features).

        Returns:
            torch.Tensor: Model output.
        """
        # Process the board state through the CNN
        cnn_out = self.cnn(board)  # Shape: (batch_size, cnn_output_size)

        # Flatten the points tensor
        points = points.view(points.size(0), -1)  # Shape: (batch_size, num_points_features)

        # Concatenate CNN output and points
        combined = torch.cat((cnn_out, points), dim=1)  # Shape: (batch_size, cnn_output_size + num_points_features)

        # Fully connected layers
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x