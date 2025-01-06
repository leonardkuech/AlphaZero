from Agent import Agent

class PrunedMinMaxAgent(Agent):
    def __init__(self, max_depth: int, player_id: int, name: str):
        super().__init__(name)
        self.max_depth = max_depth
        self.player_id = player_id
        print(f"PlayerID = {player_id}")

    def choose_move(self, game_state):
        """
        Finds the best move for the current game state using the Alpha-Beta Pruned MinMax algorithm.
        """
        move = self._find_best_move(game_state)
        return move

    def _alpha_beta(self, game_state, depth: int, alpha: float, beta: float, maximizing_player: bool):
        """
        Alpha-Beta Pruned MinMax algorithm for evaluating the game state.
        """
        if depth == 0:
            return self._evaluate_game_state(game_state)

        if game_state.check_game_over():
            winner = game_state.get_leader()
            if winner < 0:
                return 0
            return float('inf') if winner == self.player_id else float('-inf')

        if maximizing_player:
            max_eval = float('-inf')
            for move in game_state.get_moves():
                simulated_game_state = game_state.clone()
                simulated_game_state.apply_move(move)

                evaluation = self._alpha_beta(simulated_game_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for move in game_state.get_moves():
                simulated_game_state = game_state.clone()
                simulated_game_state.apply_move(move)

                evaluation = self._alpha_beta(simulated_game_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def _find_best_move(self, game_state) -> int:
        """
        Finds the best move using Alpha-Beta pruning.
        """
        best_move = float('-inf')
        best_value = float('-inf')

        for move in game_state.get_moves():
            simulated_game_state = game_state.clone()
            simulated_game_state.apply_move(move)

            move_value = self._alpha_beta(
                simulated_game_state,
                self.max_depth - 1,
                float('-inf'),
                float('inf'),
                False,
            )

            if move_value > best_value:
                best_value = move_value
                best_move = move

        print('MinMax move: ', best_move)
        return best_move

    def _evaluate_game_state(self, game_state) -> int:
        """
        Evaluates the current game state from the perspective of the player.
        """
        player_score = game_state.get_score(self.player_id)
        opponent_score = game_state.get_score(self.player_id ^ 1)
        return player_score - opponent_score