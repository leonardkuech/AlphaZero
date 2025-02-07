import math
import random
import time

from sympy.codegen.ast import float32

from Agent import Agent


class TimedMCTSAgent(Agent):
    EXPLORATION_CONSTANT = math.sqrt(2)  # UCT exploration constant

    def __init__(self, name: str, player_id: int, simulation_time=1.0):
        super().__init__(name)
        self.player_id = player_id
        self.simulation_time=simulation_time

    def choose_move(self, game_state):
        """
        Uses MCTS to choose the best move for the current game state.
        Instead of a fixed number of iterations, this version runs until
        the allocated simulation time expires.
        """
        root = Node(game_state)
        end_time = time.time() + self.simulation_time

        while time.time() < end_time:
            promising_node = self._select_promising_node(root)
            if not promising_node.game_state.check_game_over():
                if promising_node.visit_count == 0:
                    result = self._simulate_game(promising_node)
                    self._backpropagate(promising_node, result)
                else:
                    self._expand_node(promising_node)
                    node_to_simulate = promising_node.get_random_child_node()
                    result = self._simulate_game(node_to_simulate)
                    self._backpropagate(node_to_simulate, result)
            else:
                result = self._evaluate(promising_node.game_state)
                self._backpropagate(promising_node, result)

        # Uncomment the next line to print the tree for debugging purposes
        # self.print_tree(root)

        return root.get_best_move()

    def _select_promising_node(self, root_node):
        """
        Traverses the tree to find the most promising node based on UCT.
        """
        node = root_node
        while node.children:
            if node.game_state.player_to_move == self.player_id:
                node = node.get_child_with_highest_uct()
            else:
                node = node.get_child_with_inv_highest_uct()
        return node

    def _expand_node(self, node):
        """
        Expands the given node by adding all possible child nodes.
        """
        legal_moves = node.game_state.get_moves()
        for move in legal_moves:
            new_state = node.game_state.clone()
            new_state.apply_move(move)
            child_node = Node(new_state, move, node)
            node.add_child(child_node)

    def _simulate_game(self, node):
        """
        Simulates a game from the given node until it ends, returning the result.
        """
        temp_state = node.game_state.clone()
        while not temp_state.check_game_over():
            legal_moves = temp_state.get_moves()
            random_move = random.choice(legal_moves)
            temp_state.apply_move(random_move)
        return self._evaluate(temp_state)

    def _backpropagate(self, node, result):
        """
        Backpropagates the simulation result up the tree.
        """
        temp_node = node
        while temp_node is not None:
            temp_node.increment_visit_count()
            temp_node.add_score(result)
            temp_node = temp_node.parent

    def _evaluate(self, game_state):
        """
        Evaluates the final game state and returns a score for the current player.
        """
        winner = game_state.get_leader()
        if winner < 0:
            return 0.5  # Tie
        return 1.0 if winner == self.player_id else 0.0


class Node:
    def __init__(self, game_state, move=None, parent=None):
        self.game_state = game_state
        self.move = move
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.score = 0.0

    def add_child(self, child):
        self.children.append(child)

    def increment_visit_count(self):
        self.visit_count += 1

    def add_score(self, score):
        self.score += score

    def get_random_child_node(self):
        return random.choice(self.children)

    def get_child_with_highest_uct(self):
        return max(
            self.children,
            key=lambda child: self._calculate_uct(child),
        )

    def get_child_with_inv_highest_uct(self):
        return max(
            self.children,
            key=lambda child: self._calculate_inv_uct(child),
        )

    def _calculate_uct(self, child):
        if child.visit_count == 0:
            return float('inf')
        return (
            child.score / child.visit_count
            + TimedMCTSAgent.EXPLORATION_CONSTANT * math.sqrt(
                math.log(self.visit_count) / child.visit_count
            )
        )

    def _calculate_inv_uct(self, child):
        if child.visit_count == 0:
            return float('inf')
        return (
            (child.visit_count - child.score) / child.visit_count
            + TimedMCTSAgent.EXPLORATION_CONSTANT * math.sqrt(
                math.log(self.visit_count) / child.visit_count
            )
        )

    def get_best_move(self):
        best_child = max(self.children, key=lambda child: child.visit_count)
        return best_child.move