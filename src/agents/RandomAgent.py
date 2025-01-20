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
        moves = game_state.get_moves_without_passing()
        if game_state.game_started:
            self.amount_moves.append(len(moves) + 1)
        else:
            self.amount_moves.append(len(moves))

        if not moves:  # Check if the list is empty
            return float('-inf')  # Python equivalent of Integer.MIN_VALUE

        # Choose a random move from the available moves
        return random.choice(moves)