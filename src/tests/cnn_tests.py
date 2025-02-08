from CNN import GliderCNN
from Utils import INDEX_TO_MOVE

import numpy as np
import torch

from GameState import GameState


def cnn_loss_tests():
    nnet = GliderCNN(8, 9, 5)
    nnet.to('mps')
    train_data = []
    for i in range(128):
        board_tensor = torch.rand(1,8,9,9)
        player1_tensor = torch.randint(0, 5, (1, 5))
        player2_tensor = torch.randint(0, 5, (1, 5))

        policy_target = torch.softmax(torch.randn(1, len(INDEX_TO_MOVE)), dim=1)
        value_target = torch.rand(1, 1)

        train_data.append([(board_tensor, player1_tensor, player2_tensor), policy_target, value_target])

    for i in range(10000):
        nnet.trainCNN(train_data)
    # policy, value = nnet.forward(board_tensor, player1_tensor, player2_tensor)

    nnet.trainCNN(train_data)
