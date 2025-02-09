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
        return  - evaluate_game_state(game_state), -1

    if game_state.check_game_over():
        winner = game_state.get_leader()
        if winner < 0:
            return 0.0, -1
        if winner == game_state.player_to_move:
            return float('-inf'), -1
        else:
            return float('inf'), -1

    best_eval = float('-inf')
    best_move = -1

    for move in game_state.get_moves():
        simulated_game_state = game_state.clone()
        simulated_game_state.apply_move(move)

        move_value, m = minimax(simulated_game_state, depth - 1)
        if move_value > best_eval:
            best_eval = move_value
            best_move = move

    return  - best_eval, best_move


class MinMaxAgent(Agent):
    def __init__(self,name : str,  max_depth: int):
        super().__init__(name)
        self.max_depth = max_depth

    def choose_move(self, game_state : GameState) -> int:
        eval, move = minimax(game_state, self.max_depth)
        return move


