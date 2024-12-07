import random
from Agent import Agent

class GreedyAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)

    def choose_move(self, game_state):
        #TODO : implement Greedy strategy
        moves = game_state.get_moves_without_passing()

        if not moves:  # Check if the list is empty
            return float('-inf')  # Python equivalent of Integer.MIN_VALUE

        # Choose a random move from the available moves
        return random.choice(moves)