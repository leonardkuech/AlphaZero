import random
from Agent import Agent
from GameState import GameState


class GreedyAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)

    def choose_move(self, game_state : GameState):

        move = self.get_greedy_move(game_state)
        # Choose a random move from the available moves
        return move

    def get_greedy_move(self, game_state : GameState):
        moves = game_state.get_moves()

        if not moves:  # Check if the list is empty
            return float('-inf')  # Python equivalent of Integer.MIN_VALUE

        highest_value = float('-inf')
        best_move = float('-inf')

        for move in moves:
            tile = game_state.game_board.get_tile(c=move)

            if(tile.get_value() > highest_value):
                highest_value = tile.get_value()
                best_move = move

        return best_move