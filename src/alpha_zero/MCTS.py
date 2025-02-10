from numba import njit, types
import numpy as np
from numba.typed import Dict

import CNN
from GameState import GameState
from AZNode import AZNode

from Utils import INDEX_TO_MOVE, MOVE_TO_INDEX

NodeType = AZNode.class_type.instance_type

@njit(cache=True)
def uct(node : AZNode, exp: float, parent_visits : int) -> float:
    return node.q + exp * node.p * np.sqrt(parent_visits) / (1 - node.visits)

@njit(cache=True)
def inv_uct(node : AZNode, exp: float, parent_visits: int) -> float:
    return - node.q + exp * node.p * np.sqrt(parent_visits) / (1 - node.visits)

@njit(cache=True)
def backpropagate(node : AZNode, node_set , score : float):
    temp_node = node
    temp_node.update_node(score)

    while temp_node.parent >= 0:
        temp_node = node_set[temp_node.parent]
        temp_node.update_node(score)

@njit(cache=True)
def evaluate_game_state(game_state: GameState, player_id: int) -> float:
    winner = game_state.get_leader()
    if winner < 0:
        return 0.0  # Tie
    return 1.0 if winner == player_id else -1.0

@njit(cache=True)
def mask_policy(policy : np.ndarray, moves : list[int]) -> np.ndarray:
    moves_index = np.zeros(61)
    for move in moves:
        moves_index[MOVE_TO_INDEX[move]] = 1

    policy = policy * moves_index / np.sum(policy)

    print("Policy sum: ", np.sum(policy))
    return policy

class MCTS:
    def __init__(self, nnet : CNN, simulation_limit: int = 300, exp : float = np.sqrt(2)):
        self.player_id = 0
        self.index = 0
        self.simulation_limit = simulation_limit
        self.exp = exp

        self.nnet = nnet

        self.nodes = Dict.empty(key_type=types.int64, value_type=NodeType)

    def get_action_probabilities(self, game_state: GameState):

        self.player_id = game_state.player_to_move

        root = AZNode(0, game_state)
        self.nodes[self.index] = root
        self.index += 1

        for _ in range(self.simulation_limit):

            promising_node = self._select_promising_node(root)
            if not promising_node.game_state.check_game_over():

                reward = self._expand_node(promising_node)
                backpropagate(promising_node, self.nodes, reward)

            else:

                reward = evaluate_game_state(promising_node.game_state, self.player_id)
                backpropagate(promising_node, self.nodes, reward)


        self.nodes = Dict.empty(key_type=types.int64, value_type=NodeType)
        self.index = 0

        action_prob = np.zeros(61)

        for child in root.children:
            action_prob[MOVE_TO_INDEX[child.move]] = child.q

        return action_prob

    def _select_promising_node(self, node : AZNode) -> AZNode:
        while node.children:
            if node.game_state.player_to_move == self.player_id:
                node = self.get_child_with_highest_uct(node)
            else:
                node = self.get_child_with_inv_highest_uct(node)
        return node

    def _expand_node(self, node : AZNode) -> float:
        board, player1, player2 = node.game_state.encode()
        policy, value = self.nnet.predict(board, player1, player2)

        moves = node.game_state.get_moves()

        policy = mask_policy(policy, moves)

        for move in moves:
            new_state = node.game_state.clone()
            new_state.apply_move(move)

            child_node = AZNode(self.index, new_state, move, node.key, policy[MOVE_TO_INDEX[move]])
            self.nodes[self.index] = child_node
            self.index +=1

            node.add_child(child_node.key)

        if node.game_state.player_to_move == self.player_id:
            return value
        return -value

    def get_child_with_highest_uct(self, parent : AZNode) -> AZNode:
        children = [self.nodes[child] for child in parent.children]
        best_val = -1e9
        best_child = None
        for child in children:
            val = uct(child, self.exp, parent.visits)
            if val > best_val:
                best_val = val
                best_child = child
        return best_child

    def get_child_with_inv_highest_uct(self, parent : AZNode) -> AZNode:
        children = [self.nodes[child] for child in parent.children]
        best_val = -1e9
        best_child = None
        for child in children:
            val = inv_uct(child, self.exp, parent.visits)
            if val > best_val:
                best_val = val
                best_child = child
        return best_child