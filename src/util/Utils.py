import numpy as np
import math
from numba import njit, types
from numba.typed import Dict

EXPLORATION_CONSTANT = math.sqrt(2)


@njit(cache=True)
def calc_cantor(x, y) -> njit:
    x = x * 2 if x >= 0 else -x * 2 - 1
    y = y * 2 if y >= 0 else -y * 2 - 1

    return ((x + y) * (x + y + 1)) // 2 + y


@njit(cache=True)
def inverse_calc_cantor(z: int):
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    t = (w * (w + 1)) // 2
    b = z - t
    a = w - b

    def t_inv(n):
        if n % 2 == 0:
            return n // 2
        else:
            return -((n + 1) // 2)

    x = t_inv(a)
    y = t_inv(b)
    return x, y


@njit(cache=True)
def distance(start: int, end: int):
    start_x, start_y = inverse_calc_cantor(start)
    end_x, end_y = inverse_calc_cantor(end)

    if start_x - end_x != 0:
        return np.abs(start_x - end_x)
    else:
        return np.abs(start_y - end_y)


@njit(cache=True)
def in_bounds(x: int, y: int, z: int) -> bool:
    return np.abs(x) < 5 and np.abs(y) < 5 and np.abs(z) < 5


def map_move_to_index():
    mapping =  {}
    count = 0

    n = 4

    for i in range(-n, n + 1):
        r1 = max(-n, -i - n)
        r2 = min(n, -i + n)
        for j in range(r1, r2 + 1):
            if i + j + (- i - j) == 0:
                mapping[calc_cantor(i, j)] = count
                count += 1

    mapping[-1] = count

    return mapping


MOVE_TO_INDEX = map_move_to_index()
INDEX_TO_MOVE = {v: k for k, v in MOVE_TO_INDEX.items()}


@njit(cache=True)
def sum_reserve(reserve):
    total = 0
    for i in range(len(reserve)):
        total += (i + 1) * reserve[i]
    return total


@njit(cache=True)
def sum_board_values(b):
    total = 0
    for key in b:
        total += b[key]
    return total


@njit(cache=True)
def calculate_uct(visit_count: int, child_visit_count: int, child_score: int) -> float:
    if child_visit_count == 0:
        return float('inf')
    return (
            child_score / child_visit_count
            + EXPLORATION_CONSTANT * math.sqrt(
        math.log(visit_count) / child_visit_count
    )
    )


@njit(cache=True)
def calculate_inv_uct(visit_count: int, child_visit_count: int, child_score: int) -> float:
    if child_visit_count == 0:
        return float('inf')
    return (
            (child_visit_count - child_score) / child_visit_count
            + EXPLORATION_CONSTANT * math.sqrt(
        math.log(visit_count) / child_visit_count
    )
    )
