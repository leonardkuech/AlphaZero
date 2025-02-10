import torch
from sympy.unify.core import index

from Agent import Agent
from CNN import GliderCNN
from GameState import GameState
from Utils import INDEX_TO_MOVE, MOVE_TO_INDEX

import numpy as np
import logging

from MCTS import MCTS

logger = logging.getLogger(__name__)

class NNetAgent(Agent):
    def __init__(self, nnet: GliderCNN, name: str):
        super().__init__(name)
        self.nnet = nnet

    def choose_move(self, game_state: GameState):

        board, player1, player2 = game_state.encode()
        policy, value = self.nnet.predict(torch.tensor(board, dtype=torch.float32),
                                          torch.tensor(player1, dtype=torch.float32),
                                          torch.tensor(player2, dtype=torch.float32))

        valid = False
        count = 0


        while not valid:
            count += 1
            index = np.argmax(policy).item()
            move = INDEX_TO_MOVE[index]
            valid = game_state.check_move_valid(move)
            policy[0, index] = 0


        logger.debug(f'{count} moves requested')
        logger.debug(f'Selected move {move}')
        logger.debug(f'------------------------\n ')
        return move
