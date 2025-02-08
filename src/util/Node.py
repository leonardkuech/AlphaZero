import numpy as np
from numba import int32, int64, float64, types
from numba.experimental import jitclass
from numba.typed import List

from GameState import GameState
from Utils import calculate_uct, calculate_inv_uct

GameStateType = GameState.class_type.instance_type

spec = [
    ('key', int64),
    ('game_state', GameStateType),
    ('move', int32),
    ('parent', int64),
    ('children', types.ListType(int64)),  # We will assign a typed.List later.
    ('visit_count', int32),
    ('score', float64),
]

@jitclass(spec)
class Node(object):
    def __init__(self, key : int, game_state: GameState, move=-1, parent = -1):
        self.key = key
        self.game_state : GameState = game_state
        self.move = int32(move)
        self.parent : int64 = parent
        self.children = List.empty_list(int64)
        self.visit_count = 0
        self.score = 0.0

    def add_child(self, child):
        self.children.append(child)

    def increment_visit_count(self):
        self.visit_count += 1

    def add_score(self, score):
        self.score += score

    def get_random_child_node(self) -> int:
        n = len(self.children)
        if n == 0:
            return -1
        idx = np.random.randint(0, n)
        return self.children[idx]