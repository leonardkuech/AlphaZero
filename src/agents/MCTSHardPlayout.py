
from numba import njit, types
from numba.typed import Dict

from Agent import Agent
from GameState import GameState
from MinMaxAgent import minimax
from Node import Node
from Utils import calculate_uct, calculate_inv_uct

NodeType = Node.class_type.instance_type

@njit(cache=True)
def simulate_game_hard(game_state: GameState, player_id: int) -> float:
    temp_state = game_state.clone()

    while not temp_state.check_game_over():
        eval, move = minimax(temp_state, 2)
        temp_state.apply_move(move)
    return evaluate_game_state(temp_state, player_id)

@njit(cache=True)
def evaluate_game_state(game_state: GameState, player_id: int) -> float:
    winner = game_state.get_leader()
    if winner < 0:
        return 0.5  # Tie
    return 1.0 if winner == player_id else 0.0

@njit(cache=True)
def backpropagate(node: Node, node_set, result: float):
    temp_node = node

    temp_node.increment_visit_count()
    temp_node.add_score(result)

    while temp_node.parent >= 0:
        temp_node = node_set[temp_node.parent]
        temp_node.increment_visit_count()
        temp_node.add_score(result)

class MCTSAgentHardPlayout(Agent):
    SIMULATION_LIMIT = 10000  # Number of simulations per move

    def __init__(self, name: str, player_id: int):
        super().__init__(name)
        self.player_id = player_id
        self.index = 0
        self.nodes = Dict.empty(key_type=types.int64, value_type=NodeType)

    def choose_move(self, game_state: GameState):

        root = Node(0, game_state)
        self.nodes[self.index] = root
        self.index += 1

        for _ in range(self.SIMULATION_LIMIT):
            children = [self.nodes[child] for child in root.children]

            promising_node = self._select_promising_node(root)
            if not promising_node.game_state.check_game_over():
                if promising_node.visit_count == 0:
                    result = simulate_game_hard(promising_node.game_state, self.player_id)
                    backpropagate(promising_node, self.nodes, result)

                else:
                    self._expand_node(promising_node)
                    note_to_simulate = self.nodes[promising_node.get_random_child_node()]
                    result = simulate_game_hard(note_to_simulate.game_state, self.player_id)
                    backpropagate(note_to_simulate, self.nodes, result)

            else:
                result = evaluate_game_state(promising_node.game_state, self.player_id)
                backpropagate(promising_node, self.nodes, result)

        robust_move = self.get_robust_move(root)
        self.nodes = Dict.empty(key_type=types.int64, value_type=NodeType)
        self.index = 0
        return robust_move

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
        for child in children:
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_child = child
        if best_child is not None:
            return best_child.move
        return -1