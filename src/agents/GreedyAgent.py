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
        moves = game_state.get_moves_without_passing()
        highest_value =  -1
        best_move = -1

        for move in moves:
            tile = game_state.game_board[move]

            if(tile >= highest_value):
                highest_value = tile
                best_move = move

        return best_move