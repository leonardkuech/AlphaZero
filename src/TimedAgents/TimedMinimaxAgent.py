import time
import multiprocessing
import numpy as np
from numba import njit
from Agent import Agent
from GameState import GameState


@njit(cache=True)
def evaluate_game_state(game_state: GameState) -> float:
    player_score = game_state.get_score(game_state.player_to_move)
    opponent_score = game_state.get_score(game_state.player_to_move ^ 1)

    if player_score == opponent_score:
        return 0.0

    return (player_score - opponent_score) / (player_score + opponent_score)



@njit(cache=True)
def minimax(game_state: GameState, depth: int) -> (int,int):
    if depth == 0:
        return - evaluate_game_state(game_state), -1

    if game_state.check_game_over():
        winner = game_state.get_leader()
        if winner < 0:
            return 0.5, -1
        if winner == game_state.player_to_move:
            return float('-inf'), -1
        else:
            return float('inf'), -1

    best_eval = float('-inf')
    best_move = - 1

    for move in game_state.get_moves():
        simulated_game_state = game_state.clone()
        simulated_game_state.apply_move(move)

        move_value, m = minimax(simulated_game_state, depth - 1)
        if move_value > best_eval:
            best_eval = move_value
            best_move = move

    return - best_eval, best_move

def minimax_worker(game_state, depth, queue):
    result = minimax(game_state, depth)
    queue.put(result)

class TimedMinimaxAgent(Agent):
    def __init__(self, name: str, time_limit: float):
        super().__init__(name)
        self.time_limit = time_limit

    def choose_move(self, game_state: GameState) -> int:
        start_time = time.time()

        eval, last_best_move = minimax(game_state, 2)

        depth = 2

        try:
            multiprocessing.set_start_method('fork')
        except RuntimeError:
            pass

        while True:
            elapsed = time.time() - start_time
            remaining_time = self.time_limit - elapsed
            if remaining_time <= 0:
                break  # Time is up

            queue = multiprocessing.Queue()
            cloned_state = game_state.clone()
            p = multiprocessing.Process(target=minimax_worker, args=(cloned_state, depth, queue))
            p.start()

            p.join(remaining_time)

            if p.is_alive():
                p.terminate()
                p.join()
                break

            try:
                eval_value, move = queue.get()
                if move is not None and move != np.int32(-1):
                    last_best_move = move
            except Exception:
                break

            depth += 1

        return int(last_best_move)