import logging
import math
from re import search

import numpy as np
import torch

from CNN import GliderCNN as cnn
from GameState import GameState, MOVE_TO_INDEX

log = logging.getLogger(__name__)


def evaluate(game_state: GameState):
    """
    Evaluates the final game state and returns a score for the current player.
    """
    winner = game_state.get_leader()
    if winner < 0:
        return 0.5  # Tie
    return 1.0 if winner == game_state.player_to_move else 0


class MCTS():
    SIMULATION_LIMIT = 10000  # Number of simulations per move
    EXPLORATION_CONSTANT = math.sqrt(2)  # UCT exploration constant

    def __init__(self, nnet: cnn):
        self.nnet = nnet
        self.Qsa = {}  # Expected reward taking an action a from a GameState s
        self.Nsa = {}  # Number of times action a was taken from GameState s
        self.Ns = {} # Number of times board s was visited
        self.Ps = {}  # Stores initial policy from neural net

    def get_action_probabilities(self, game_state: GameState):

        for i in range(MCTS.SIMULATION_LIMIT):
            self.search(game_state)

        probabilities = torch.zeros(len(MOVE_TO_INDEX))

        s = game_state.string_representation()

        for move in game_state.get_moves():
            probabilities[MOVE_TO_INDEX[move]] = self.Nsa[(s,move)] / (self.Ns[s] - 1)

        return probabilities


    def search(self, game_state: GameState):

        if game_state.check_game_over(): return 1 - evaluate(game_state)

        s = game_state.string_representation()

        if s not in self.Ns:
            self.Ns[s] = 1
            board, player1, player2 = game_state.encode()
            with torch.inference_mode():
                self.Ps[s], v = self.nnet.forward(board, player1, player2)
            return 1 - v


        max_u, best_a = -float("inf"), -float("inf")
        for a in game_state.get_moves():
            u = self.Qsa[(s,a)] + MCTS.EXPLORATION_CONSTANT * self.Ps[s][MOVE_TO_INDEX[a]] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s,a)])
            if u > max_u:
                max_u = u
                best_a = a
        a = best_a

        next_state = game_state.clone()
        next_state.apply_move(a)

        v = self.search(next_state)

        self.Ns[s] += 1
        self.Nsa[(s,a)] += 1
        self.Qsa[(s,a)] = (self.Nsa[(s,a)] * self.Qsa[(s,a)] + v) / (self.Nsa[(s,a)])
        return 1 - v