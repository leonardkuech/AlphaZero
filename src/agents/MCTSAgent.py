import math

from numba import njit
import numpy as np
from torch import read_vitals

from Agent import Agent
from GameState import GameState
from Node import Node
from Utils import calculate_uct, calculate_inv_uct


@njit(cache=True)
def simulate_game(game_state: GameState, player_id: int) -> float:
    temp_state = game_state.clone()
    while not temp_state.check_game_over():
        legal_moves = temp_state.get_moves()
        index = np.random.randint(0, len(legal_moves))
        random_move = legal_moves[index]
        temp_state.apply_move(random_move)
    return evaluate_game_state(temp_state, player_id)

@njit(cache=True)
def evaluate_game_state(game_state: GameState, player_id: int) -> float:
    winner = game_state.get_leader()
    if winner < 0:
        return 0.5  # Tie
    return 1.0 if winner == player_id else 0.0

class MCTSAgent(Agent):
    SIMULATION_LIMIT = 30000  # Number of simulations per move

    def __init__(self, name: str, player_id: int):
        super().__init__(name)
        self.player_id = player_id
        self.index = 0
        self.nodes : dict[int, Node]= {}

    def choose_move(self, game_state: GameState):

        root = Node(0, game_state)
        self.nodes[self.index] = root
        self.index += 1

        for _ in range(self.SIMULATION_LIMIT):
            promising_node = self._select_promising_node(root)
            if not promising_node.game_state.check_game_over():
                if promising_node.visit_count == 0:
                    result = simulate_game(promising_node.game_state, self.player_id)
                    self._backpropagate(promising_node, result)

                else:
                    self._expand_node(promising_node)
                    note_to_simulate = self.nodes[promising_node.get_random_child_node()]
                    result = simulate_game(note_to_simulate.game_state, self.player_id)
                    self._backpropagate(note_to_simulate, result)

            else:
                result = evaluate_game_state(promising_node.game_state, self.player_id)
                self._backpropagate(promising_node, result)
        print(self.get_robust_move(root))
        return self.get_robust_move(root)

    def _select_promising_node(self, node : Node) -> Node:
        while node.children:
            if node.game_state.player_to_move == self.player_id:
                node = self.get_child_with_highest_uct(node)
            else:
                node = self.get_child_with_inv_highest_uct(node)
        return node

    def _expand_node(self, node : Node):
        legal_moves = node.game_state.get_moves()

        for move in legal_moves:
            new_state = node.game_state.clone()
            new_state.apply_move(move)

            child_node = Node(self.index, new_state, move, node.key)
            self.nodes[self.index] = child_node
            self.index +=1

            node.add_child(child_node.key)

    def _backpropagate(self, node : Node, result : float):
        temp_node = node

        temp_node.increment_visit_count()
        temp_node.add_score(result)

        while temp_node.parent >= 0:
            temp_node = self.nodes[temp_node.parent]
            temp_node.increment_visit_count()
            temp_node.add_score(result)

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

    def get_robust_move(self, node):
        children = [self.nodes[child] for child in node.children]
        best_visits = -1
        best_child = None

        print([(child.visit_count, child.move) for child in children])

        for child in children:
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_child = child
        if best_child is not None:
            return best_child.move
        return -1