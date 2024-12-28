from Agent import Agent
from CNN import GliderCNN
from GameState import GameState

import numpy as np


class NNetAgent(Agent):
    def __init__(self, nnet: GliderCNN, name: str):
        super().__init__(name)
        self.nnet = nnet

    def choose_action(self, game_state: GameState):
        policy, value = self.nnet.predict(game_state.encode())
        action = np.argmax(policy)
        return action
