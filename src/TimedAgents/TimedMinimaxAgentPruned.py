import time
import multiprocessing
import numpy as np
from numba import njit
from Agent import Agent
from GameState import GameState


# -----------------------------------------------------------
# New minimax with alpha-beta pruning (negamax style)
@njit(cache=True)
def minimax_ab(game_state: GameState, depth: int, alpha: float, beta: float):
    # Terminal condition: depth 0 or game over.
    if depth == 0:
        # Return the evaluation from the perspective of the player to move.
        return -evaluate_game_state(game_state), np.int32(-1)

    if game_state.check_game_over():
        winner = game_state.get_leader()
        if winner < 0:
            return 0.0, np.int32(-1)
        if winner == game_state.player_to_move:
            return float('-inf'), np.int32(-1)
        else:
            return float('inf'), np.int32(-1)

    best_eval = float('-inf')
    best_move = np.int32(-1)

    moves = game_state.get_moves()
    for move in moves:
        simulated_state = game_state.clone()
        simulated_state.apply_move(move)
        # Recursive call with swapped alpha and beta values.
        value, _ = minimax_ab(simulated_state, depth - 1, -beta, -alpha)
        # Negate the value because of negamax.
        value = -value

        if value > best_eval:
            best_eval = value
            best_move = move

        # Update alpha.
        if value > alpha:
            alpha = value

        # Beta cutoff.
        if alpha >= beta:
            break

    return -best_eval, np.int32(best_move)


# -----------------------------------------------------------
# The evaluation function remains unchanged.
@njit(cache=True)
def evaluate_game_state(game_state: GameState) -> int:
    player_score = game_state.get_score(game_state.player_to_move)
    opponent_score = game_state.get_score(game_state.player_to_move ^ 1)
    return player_score - opponent_score


# -----------------------------------------------------------
# The worker now calls minimax_ab.
def minimax_worker(game_state, depth, queue):
    result = minimax_ab(game_state, depth, float('-inf'), float('inf'))
    queue.put(result)


# -----------------------------------------------------------
# TimedMinimaxAgent using iterative deepening with alpha-beta pruning.
class TimedMinimaxAgent(Agent):
    def __init__(self, name: str, time_limit: float):
        super().__init__(name)
        self.time_limit = time_limit  # allowed time in seconds

    def choose_move(self, game_state: GameState) -> int:
        start_time = time.time()

        # Warm-up call at shallow depth to compile the function.
        eval_value, last_best_move = minimax_ab(game_state, 2, float('-inf'), float('inf'))
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
                if move is not None and move != np.int32(-1):
                    last_best_move = move
            except Exception:
                break

            depth += 1

        return int(last_best_move)