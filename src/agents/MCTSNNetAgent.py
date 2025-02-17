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

class MCTSNNetAgent(Agent):
    def __init__(self, nnet, name: str):
        super().__init__(name)
        self.mcts = MCTS(nnet=nnet, simulation_limit=100)

    def choose_move(self, game_state: GameState):

        prop = self.mcts.get_action_probabilities(game_state)

        valid = False
        count = 0


        while not valid:
            count += 1
            index = np.argmax(prop).item()
            move = INDEX_TO_MOVE[index]
            valid = game_state.check_move_valid(move)
            prop[index] = 0


        logger.debug(f'{count} moves requested')
        logger.debug(f'Selected move {move}')
        logger.debug(f'------------------------\n ')
        return move
