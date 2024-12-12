from Agent import Agent


def _evaluate_game_state(game_state) -> int:
    """
    Evaluates the current game state from the perspective of the player.
    """
    player_score = game_state.get_score(game_state.get_player_id_to_move())
    opponent_score = game_state.get_score(game_state.get_player_id_to_move() ^ 1)
    return player_score - opponent_score


class MinMaxAgent(Agent):
    def __init__(self, max_depth: int, player_id: int, name: str):
        super().__init__(name)
        self.max_depth = max_depth
        self.player_id = player_id
        print(f"PlayerID = {player_id}")

    def choose_move(self, game_state):
        """
        Finds the best move for the current game state using the MinMax algorithm.
        """
        return self._find_best_move(game_state)

    def _min_max(self, game_state, depth: int):
        """
        Recursive MinMax algorithm to evaluate the game state.
        """
        if depth == 0:
            return - _evaluate_game_state(game_state)

        if game_state.check_game_over():
            winner = game_state.get_leader()
            if winner < 0:
                return 0
            return (
                float('-inf') if winner == game_state.get_player_id_to_move() else float('inf')
            )

        best_eval = float('-inf')

        for move in game_state.get_moves():
            simulated_game_state = game_state.clone()
            simulated_game_state.apply_move(move)
            simulated_game_state.next_player_to_move()

            move_value = self._min_max(simulated_game_state, depth - 1)
            best_eval = max(best_eval, move_value)

        return -best_eval

    def _find_best_move(self, game_state) -> int:
        """
        Finds the best move using the MinMax algorithm.
        """
        best_move = -1
        best_value = float('-inf')

        for move in game_state.get_moves():
            simulated_game_state = game_state.clone()
            simulated_game_state.apply_move(move)
            simulated_game_state.next_player_to_move()

            move_value = self._min_max(simulated_game_state, self.max_depth - 1)

            if move_value > best_value:
                best_value = move_value
                best_move = move

        return best_move

