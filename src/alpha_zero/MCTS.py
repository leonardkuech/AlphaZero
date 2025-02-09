import logging
import math
import torch
from numba import njit, types
import numpy as np
from numba.typed import Dict

from Node import Node

from CNN import GliderCNN as cnn
from GameState import GameState
from Utils import MOVE_TO_INDEX, calculate_uct, calculate_inv_uct

NodeType = Node.class_type.instance_type

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

EPS = 1e-8

@njit(cache=True)
def evaluate(game_state: GameState):
    winner = game_state.get_leader()
    if winner < 0:
        return 0.0  # Tie
    return 1.0 if winner == game_state.player_to_move else 0.0

@njit(cache=True)
def backpropagate(node : Node, node_set , result : float):
    temp_node = node

    temp_node.increment_visit_count()
    temp_node.add_score(result)

    while temp_node.parent >= 0:
        temp_node = node_set[temp_node.parent]
        temp_node.increment_visit_count()
        temp_node.add_score(result)

class MCTS():

    def __init__(self, nnet: cnn, exp_const = 1, simulation_limit = 150):
        self.nnet = nnet
        self.SIMULATION_LIMIT = simulation_limit
        self.EXP_CONST = exp_const

        self.nodes = Dict.empty(key_type=types.int64, value_type=NodeType)
        index : int = 0

    def get_action_probabilities(self, game_state: GameState):

        for i in range(self.SIMULATION_LIMIT):
            self.search(game_state)

        probabilities = torch.zeros(len(MOVE_TO_INDEX))

        s = game_state.string_representation()
        sum = 0
        for move in game_state.get_moves():
            sum += self.Nsa.get((s, move),0)
            probabilities[MOVE_TO_INDEX[move]] = self.Nsa.get((s, move),0) / (self.Ns[s])

        return probabilities


    def get_child_with_highest_uct(self, parent : Node) -> Node:
        children = [self.nodes[child] for child in parent.children]
        best_val = -1e9
        best_child = None
        for child in children:
            val = calculate_uct(parent.visit_count, child.visit_count, child.score)
            if val > best_val:
                best_val = val
                best_child = child
        return best_child

    def get_child_with_inv_highest_uct(self, parent : Node) -> Node:
        children = [self.nodes[child] for child in parent.children]
        best_val = -1e9
        best_child = None
        for child in children:
            val = calculate_inv_uct(parent.visit_count, child.visit_count, child.score)
            if val > best_val:
                best_val = val
                best_child = child
        return best_child