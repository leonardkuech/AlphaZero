import random
from Agent import Agent

class RandomAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.amount_moves = []


    def choose_move(self, game_state):
        """
        Selects a random move from the available moves in the game state.
        If no moves are available, returns a very small value (equivalent to Integer.MIN_VALUE in Java).
        """
        moves = game_state.get_moves()

        # Choose a random move from the available moves
        return random.choice(moves)