import time
import math
import numpy as np
from numba import njit

from Agent import Agent
from GameState import GameState
from Node import Node

EXPLORATION_CONSTANT = np.sqrt(2)  # UCT exploration constant


@njit
def simulate_game(game_state: GameState, player_id: int) -> float:
    temp_state = game_state.clone()
    while not temp_state.check_game_over():
        legal_moves = temp_state.get_moves()
        index = np.random.randint(0, len(legal_moves))
        random_move = legal_moves[index]
        temp_state.apply_move(random_move)
    return evaluate_game_state(temp_state, player_id)


@njit
def evaluate_game_state(game_state: GameState, player_id: int) -> float:
    winner = game_state.get_leader()
    if winner < 0:
        return 0.5  # Tie
    return 1.0 if winner == player_id else 0.0


class TimedMCTSAgent(Agent):
    def __init__(self, name: str, player_id: int, time_limit: float = 1.0):
        super().__init__(name)
        self.player_id = player_id
        self.time_limit = time_limit

    def choose_move(self, game_state: GameState):
        root = Node(game_state)
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            promising_node = self._select_promising_node(root)

            if not promising_node.game_state.check_game_over():
                if promising_node.visit_count == 0:
                    result = simulate_game(promising_node.game_state, self.player_id)
                    self._backpropagate(promising_node, result)
                else:
                    self._expand_node(promising_node)
                    node_to_simulate = promising_node.get_random_child_node()
                    result = simulate_game(node_to_simulate.game_state, self.player_id)
                    self._backpropagate(node_to_simulate, result)
            else:
                result = evaluate_game_state(promising_node.game_state, self.player_id)
                self._backpropagate(promising_node, result)

        return root.get_best_move()

    def _select_promising_node(self, root_node):
        node = root_node
        while node.children:
            # Choose child based on UCT values.
            if node.game_state.player_to_move == self.player_id:
                node = node.get_child_with_highest_uct()
            else:
                node = node.get_child_with_inv_highest_uct()
        return node

    def _expand_node(self, node):
        legal_moves = node.game_state.get_moves()
        for move in legal_moves:
            new_state = node.game_state.clone()
            new_state.apply_move(move)
            child_node = Node(new_state, move, node)
            node.add_child(child_node)

    def _backpropagate(self, node, result):
        temp_node = node
        while temp_node is not None:
            temp_node.increment_visit_count()
            temp_node.add_score(result)
            temp_node = temp_node.parent