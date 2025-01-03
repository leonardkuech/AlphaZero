import torch
import torch.nn as nn

from GameState import INDEX_TO_MOVE


class GliderCNN(nn.Module):
    def __init__(self, num_channels=8, grid_size=9, num_features_per_player=5):
        """
        Args:
            num_channels (int): Number of input channels for the board representation.
            grid_size (int): Size of the board (assumed square, grid_size x grid_size).
            num_features_per_player (int): Length of the feature tensor for each player.
        """
        super(GliderCNN, self).__init__()
        self.device = torch.device('mps')

        # CNN for processing the board state
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the flattened output size of the CNN
        self.cnn_output_size = 64 * grid_size * grid_size

        # Fully connected layers for shared representation
        total_features = self.cnn_output_size + (2 * num_features_per_player)
        self.fc_shared = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU()
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Outputs a scalar between 0 and 1
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(INDEX_TO_MOVE)),
            nn.Softmax(dim=1)  # Outputs probabilities for all possible moves
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, board, player1_features, player2_features):
        # Process the board state through the CNN
        cnn_out = self.cnn(board)  # Shape: (batch_size, cnn_output_size)

        # Concatenate player features with CNN output
        combined = torch.cat((cnn_out, player1_features, player2_features), dim=1)

        # Fully connected layers
        x = self.fc_shared(combined)

        return self.policy_head(x), self.value_head(x)

    def predict(self, board, player1, player2):
        with torch.inference_mode():
            return self.forward(board, player1, player2)
        #Todo cache

    def trainCNN(self, trainingExamples):
        self.train()

        # Separate the data into individual tensors
        inputs, policy_targets, value_targets = zip(*trainingExamples)

        # Unpack the inputs tuple for better clarity
        board_tensors, player1_tensors, player2_tensors = zip(*inputs)

        board_tensors = torch.cat(board_tensors)
        player1_tensors = torch.cat(player1_tensors)
        player2_tensors = torch.cat(player2_tensors)
        policy_targets = torch.cat(policy_targets)
        value_targets = torch.cat(value_targets)

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(
            board_tensors, player1_tensors, player2_tensors, policy_targets, value_targets
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        for board, player1, player2, policy, value in loader:
            # Move tensors to the correct device
            board = board.to(self.device)
            player1 = player1.to(self.device)
            player2 = player2.to(self.device)
            policy = policy.to(self.device)
            value = value.to(self.device)

            self.optimizer.zero_grad()
            # Forward pass
            predicted_policy, predicted_value = self.forward(board, player1, player2)
            # Compute loss
            loss = self.loss(predicted_policy, predicted_value, policy, value)
            print(loss.item())
            # Backpropagation
            loss.backward()
            self.optimizer.step()

        self.eval()

    def loss(self, pred_policy, pred_value, target_policy, target_value):

        value_loss = (pred_value - target_value).pow(2)

        print('value : ', value_loss.mean().item())

        epsilon = 1e-7

        policy_loss = - (target_policy * torch.log(pred_policy + epsilon)).sum(dim=1)
        print('policy : ' , policy_loss.mean().item())

        total_loss = value_loss + policy_loss

        return total_loss.mean()