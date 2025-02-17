import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from schedulefree import AdamWScheduleFree

from Utils import INDEX_TO_MOVE

logger = logging.getLogger(__name__)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        downsample=None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activation = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

class GliderCNN(nn.Module):
    def __init__(self, num_channels=8, grid_size=9, num_features_per_player=5, channels_inner=24):
        """
        Args:
            num_channels (int): Number of input channels for the board representation.
            grid_size (int): Size of the board (assumed square, grid_size x grid_size).
            num_features_per_player (int): Length of the feature tensor for each player.
        """
        super(GliderCNN, self).__init__()
        self.device = torch.device('cpu')

        # CNN for processing the board state

        self.cnn1 = nn.Sequential(
            nn.Conv2d(num_channels, channels_inner, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_inner),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList()

        for _ in range(1):
            self.blocks.append(
                BasicBlock(
                    inplanes=channels_inner,
                    planes=channels_inner,
                )
            )

        self.blocks.append(
            BasicBlock(
                inplanes=channels_inner,
                planes=channels_inner * 2,
                downsample=conv1x1(channels_inner, channels_inner * 2)
            )
        )

        for _ in range(1):
            self.blocks.append(
                BasicBlock(
                    inplanes=channels_inner * 2,
                    planes=channels_inner * 2,
                )
            )

        # Calculate the flattened output size of the CNN
        self.cnn_output_size = channels_inner * 2 * grid_size * grid_size

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        total_features = channels_inner * 2

        # Player feature projection
        self.in_projection = nn.Sequential(
            nn.BatchNorm1d(2 * num_features_per_player),
            nn.Linear(2 * num_features_per_player, total_features),
            nn.BatchNorm1d(total_features),
            nn.SiLU(),
        )

        # Fully connected layers for shared representation
        self.fc_shared = nn.Sequential(
            nn.Linear(total_features, 100),
            nn.BatchNorm1d(100),
            nn.SiLU(),
            nn.Dropout(p=0.1)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(100, 1),
            nn.Tanh()  # Outputs a scalar between -1 and 1
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(100, len(INDEX_TO_MOVE)),
            nn.Softmax(dim=1)  # Outputs probabilities for all possible moves
        )

        self.optimizer = AdamWScheduleFree(params=self.parameters(), lr=0.05, warmup_steps=100)
        self.to(self.device)
        self.eval()
        self.optimizer.eval()

    def forward(self, board, player1_features, player2_features):
        # Process the board state through the CNN
        cnn_out = self.cnn1(board)  # Shape: (batch_size, cnn_output_size)

        for block in self.blocks:
            cnn_out = block(cnn_out)

        cnn_out = self.avg_pool(cnn_out)
        cnn_out = self.flatten(cnn_out)

        player_features = torch.cat((player1_features, player2_features), dim=1)
        player_features = self.in_projection(player_features)

        # Concatenate player features with CNN output
        combined = cnn_out + player_features

        # Fully connected layers
        x = self.fc_shared(combined)

        return self.policy_head(x), self.value_head(x)

    @torch.compile
    def predict(self, board, player1, player2):
        with torch.inference_mode():
            board = board.to(self.device)
            player1 = player1.to(self.device)
            player2 = player2.to(self.device)
            p, v = self.forward(board, player1, player2)

            return p.detach().numpy(), v.detach().numpy()

    def trainCNN(self, trainingExamples):

        updated_nnet = copy.deepcopy(self)
        updated_nnet.load_state_dict(self.state_dict())
        updated_nnet.to(self.device)
        updated_nnet = torch.compile(updated_nnet)

        updated_nnet.train()
        updated_nnet.optimizer.train()

        # Separate the data into individual tensors
        inputs, policy_targets, value_targets = zip(*trainingExamples)

        # Unpack the inputs tuple for better clarity
        board_tensors, player1_tensors, player2_tensors = zip(*inputs)

        board_tensors = np.squeeze(board_tensors, axis=1)
        player1_tensors = np.squeeze(player1_tensors, axis=1)
        player2_tensors = np.squeeze(player2_tensors, axis=1)

        policy_targets = np.squeeze(policy_targets, axis=1)
        value_targets = np.squeeze(value_targets, axis=1)

        board_tensors = torch.tensor(board_tensors).to(self.device)
        player1_tensors = torch.tensor(player1_tensors).to(self.device)
        player2_tensors = torch.tensor(player2_tensors).to(self.device)

        policy_targets = torch.tensor(policy_targets).to(self.device)
        value_targets = torch.tensor(value_targets).to(self.device)

        # Create TensorDataset
        dataset = torch.utils.data.TensorDataset(
            board_tensors, player1_tensors, player2_tensors, policy_targets, value_targets
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

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
            predicted_policy, predicted_value = updated_nnet.forward(board, player1, player2)

            loss = updated_nnet.loss(predicted_policy, predicted_value, policy, value)
            loss_total += loss.item()
            acc =  (torch.argmax(policy, dim=1) == torch.argmax(predicted_policy, dim=1)).sum() / len(policy)
            acc_total += acc

            loss.backward()
            updated_nnet.optimizer.step()

        acc_total = acc_total / len(loader)
        loss_total = loss_total / len(loader)
        logger.info(f'Pol Acc = {acc_total}')
        logger.info(f'Loss is {loss_total}')
        updated_nnet.eval()
        updated_nnet.optimizer.eval()

        return updated_nnet

    def loss(self, pred_policy, pred_value, target_policy, target_value):

        value_loss = (pred_value - target_value).pow(2)

        epsilon = 1e-7

        policy_loss = - (target_policy * torch.log(pred_policy + epsilon)).sum(dim=1)

        total_loss = value_loss + policy_loss

        return total_loss.mean()