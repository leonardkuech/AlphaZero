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
        logger.debug(f'------------------------\n')


        mcts = MCTS(self.nnet, exp_const=0.1, simulations=25)
        prob = mcts.get_action_probabilities(game_state)

        logger.debug(f'Prob: {prob}')

        valid = False
        count = 0
        index = -1

        while not valid:
            count += 1
            #logger.debug(f'{policy} chose move {np.argmax(policy).item()}')
            index = np.argmax(prob).item()
            move = INDEX_TO_MOVE[index]
            valid = game_state.check_move_valid(move)
            prob[index] = 0


        logger.debug(f'{count} moves requested')
        logger.debug(f'Selected move {move}')
        logger.debug(f'------------------------\n ')
        return move
