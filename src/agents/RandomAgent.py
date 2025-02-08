import random
from Agent import Agent

class RandomAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.amount_moves = []


    def choose_move(self, game_state):
        moves = game_state.get_moves()
        return random.choice(moves)