import logging

import torch
import torch.nn as nn

from Utils import INDEX_TO_MOVE

logger = logging.getLogger(__name__)
class GliderCNN(nn.Module):
    def __init__(self, num_channels=8, grid_size=9, num_features_per_player=5):
        """
        Args:
            num_channels (int): Number of input channels for the board representation.
            grid_size (int): Size of the board (assumed square, grid_size x grid_size).
            num_features_per_player (int): Length of the feature tensor for each player.
        """
        super(GliderCNN, self).__init__()
        self.device = torch.device('cpu')

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
            nn.Tanh()  # Outputs a scalar between -1 and 1
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(INDEX_TO_MOVE)),
            nn.Softmax(dim=1)  # Outputs probabilities for all possible moves
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.to(self.device)

    def forward(self, board, player1_features, player2_features):
        # Process the board state through the CNN
        cnn_out = self.cnn(board)  # Shape: (batch_size, cnn_output_size)

        # Concatenate player features with CNN output
        combined = torch.cat((cnn_out, player1_features, player2_features), dim=1)

        # Fully connected layers
        x = self.fc_shared(combined)

        return self.policy_head(x), self.value_head(x)

    @torch.compile
    def predict(self, board, player1, player2):
        with torch.inference_mode():
            board = board.to(self.device)
            player1 = player1.to(self.device)
            player2 = player2.to(self.device)
            return self.forward(board, player1, player2)
        #Todo cache

    def trainCNN(self, trainingExamples):

        updated_nnet = self.__class__()  # Assumes your model class can be instantiated without arguments
        updated_nnet.load_state_dict(self.state_dict())  # Copy model parameters
        updated_nnet.to(self.device)  # Move the new model to the correct device
        updated_nnet = torch.compile(updated_nnet)

        updated_nnet.train()

        # Separate the data into individual tensors
        inputs, policy_targets, value_targets = zip(*trainingExamples)

        # Unpack the inputs tuple for better clarity
        board_tensors, player1_tensors, player2_tensors = zip(*inputs)

        board_tensors = torch.tensor(board_tensors)
        print(board_tensors.shape)
        print(board_tensors)
        player1_tensors = torch.tensor(player1_tensors)
        player2_tensors = torch.tensor(player2_tensors)

        board_tensors = torch.cat(board_tensors).to(self.device)
        player1_tensors = torch.cat(player1_tensors).to(self.device)
        player2_tensors = torch.cat(player2_tensors).to(self.device)
        policy_targets = torch.cat(policy_targets).to(self.device)
        value_targets = torch.cat(value_targets).to(self.device)

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(
            board_tensors, player1_tensors, player2_tensors, policy_targets, value_targets
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        acc_total = 0
        loss_total = 0

        for board, player1, player2, policy, value in loader:
            board, player1, player2, policy, value = (
                board.to(self.device),
                player1.to(self.device),
                player2.to(self.device),
                policy.to(self.device),
                value.to(self.device),
            )

            updated_nnet.optimizer.zero_grad()
            # Forward pass
            predicted_policy, predicted_value = updated_nnet.forward(board, player1, player2)
            # Compute loss
            loss = updated_nnet.loss(predicted_policy, predicted_value, policy, value)
            loss_total += loss.item()
            acc =  (torch.argmax(policy, dim=1) == torch.argmax(predicted_policy, dim=1)).sum() / len(policy)
            acc_total += acc
            # Backpropagation
            loss.backward()
            updated_nnet.optimizer.step()

        acc_total = acc_total / len(loader)
        loss_total = loss_total / len(loader)
        logger.info(f'Pol Acc = {acc_total}')
        logger.info(f'Loss is {loss_total}')
        updated_nnet.eval()

        return updated_nnet

    def loss(self, pred_policy, pred_value, target_policy, target_value):

        value_loss = (pred_value - target_value).pow(2)

        epsilon = 1e-7

        policy_loss = - (target_policy * torch.log(pred_policy + epsilon)).sum(dim=1)

        total_loss = value_loss + policy_loss

        return total_loss.mean()