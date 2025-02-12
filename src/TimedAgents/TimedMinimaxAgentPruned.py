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
def minimax_ab(game_state: GameState, depth: int, alpha: float, beta: float, maximizing: bool) -> (float, int):
    if depth == 0:
        if maximizing:
            return evaluate_game_state(game_state), -1
        else:
            return  - evaluate_game_state(game_state), -1

    if game_state.check_game_over():
        winner = game_state.get_leader()
        if winner < 0:
            return 0.0, -1
        if winner == game_state.player_to_move:
            if maximizing:
                return float('inf'), -1
            else:
                return float('-inf'), -1
        else:
            if maximizing:
                return float('-inf'), -1
            else:
                return float('inf'), -1

    if maximizing:
        best_eval = float('-inf')
        best_move = -1

        moves = game_state.get_moves()
        for move in moves:
            simulated_state = game_state.clone()
            simulated_state.apply_move(move)
            # Recurse as a minimizing node.
            value, _ = minimax_ab(simulated_state, depth - 1, alpha, beta, False)
            if value > best_eval:
                best_eval = value
                best_move = move
            if best_eval > alpha:
                alpha = best_eval
            if beta <= alpha:
                break
        return best_eval, best_move

    else:

        best_eval = float('inf')
        best_move = -1
        for move in game_state.get_moves():
            simulated_state = game_state.clone()
            simulated_state.apply_move(move)
            value, _ = minimax_ab(simulated_state, depth - 1, alpha, beta, True)
            if value < best_eval:
                best_eval = value
                best_move = move
            if best_eval < beta:
                beta = best_eval
            if beta <= alpha:
                break  # Alpha cutoff.
        return best_eval, best_move

def minimax_worker(game_state, depth, queue):
    result = minimax_ab(game_state, depth, float('-inf'), float('inf'), True)
    queue.put(result)

class TimedMinimaxAgentPruned(Agent):
    def __init__(self, name: str, time_limit: float):
        super().__init__(name)
        self.time_limit = time_limit
        self.depths = []

    def choose_move(self, game_state: GameState) -> int:
        start_time = time.time()

        # Warm-up: compile the minimax_ab function at a shallow depth.
        eval_value, last_best_move = minimax_ab(game_state, 2, float('-inf'), float('inf'), True)
        depth = 2

        try:
            multiprocessing.set_start_method('fork')
        except RuntimeError:
            pass

        while True:
            elapsed = time.time() - start_time
            remaining_time = self.time_limit - elapsed
            if remaining_time <= 0:
                break

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
                if move is not None and move != -1:
                    last_best_move = move
            except Exception:
                break

            depth += 1

        self.depths.append(depth)
        return int(last_best_move)