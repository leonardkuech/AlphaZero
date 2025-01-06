from Agent import Agent
from CNN import GliderCNN
from GameState import GameState, INDEX_TO_MOVE

import numpy as np
import logging

logger = logging.getLogger(__name__)

class NNetAgent(Agent):
    def __init__(self, nnet: GliderCNN, name: str):
        super().__init__(name)
        self.nnet = nnet

    def choose_move(self, game_state: GameState):
        board, player1, player2 = game_state.encode()
        policy, value = self.nnet.predict(board, player1, player2)
        policy = policy.clone()
        valid = False


        while not valid:
            #logger.info(f'{policy} chose move {np.argmax(policy).item()}')
            index = np.argmax(policy).item()
            move = INDEX_TO_MOVE[index]
            valid = game_state.check_move_valid(move)
            policy[0, index] = 0

        return move
