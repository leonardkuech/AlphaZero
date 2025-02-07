import time
from Agent import Agent

def _evaluate_game_state(game_state) -> int:
    """
    Evaluates the current game state from the perspective of the player.
    """
    player_score = game_state.get_score(game_state.get_player_id_to_move())
    opponent_score = game_state.get_score(game_state.get_player_id_to_move() ^ 1)
    return player_score - opponent_score

class Timeout(Exception):
    """Custom exception for signaling that the time limit has been reached."""
    pass

class TimedMinimaxAgent(Agent):
    def __init__(self, player_id: int, name: str, time_limit: float = 1.0):
        """
        Initialize the MinMax agent.

        Args:
            max_depth (int): An optional maximum search depth.
            player_id (int): The agent's player ID.
            name (str): The agent's name.
            time_limit (float): The maximum time (in seconds) allowed for choosing a move.
        """
        super().__init__(name)
        self.player_id = player_id
        self.time_limit = time_limit  # in seconds

    def choose_move(self, game_state):
        """
        Uses iterative deepening with the MinMax algorithm under a time constraint to find the best move.
        Iterative deepening is used so that if time runs out, the best move found at the previous depth is returned.
        """
        start_time = time.time()
        best_move = None
        current_depth = 1

        # Iterative deepening loop: increase depth until time expires (or maximum depth reached)
        while True:
            try:
                # Search at the current depth.
                move = self._find_best_move(game_state, current_depth, start_time)
                best_move = move
                current_depth += 1
            except Timeout:
                # Time limit reached during this iteration.
                break

            # Also check the time before starting a new deeper iteration.
            if time.time() - start_time >= self.time_limit:
                break

        return best_move

    def _min_max(self, game_state, depth: int, start_time):
        # Check the time limit.
        if time.time() - start_time >= self.time_limit:
            raise Timeout("Time limit reached during search")

        # Terminal condition: if at maximum depth or the game is over.
        if depth == 0:
            return -_evaluate_game_state(game_state)

        if game_state.check_game_over():
            winner = game_state.get_leader()
            if winner < 0:
                return 0
            # In negamax style, return negative infinity if the current player is the winner,
            # positive infinity otherwise.
            return float('-inf') if winner == game_state.get_player_id_to_move() else float('inf')

        best_eval = float('-inf')
        for move in game_state.get_moves():
            # Check the time limit before exploring further.
            if time.time() - start_time >= self.time_limit:
                raise Timeout("Time limit reached during search")
            simulated_game_state = game_state.clone()
            simulated_game_state.apply_move(move)
            move_value = self._min_max(simulated_game_state, depth - 1, start_time)
            best_eval = max(best_eval, move_value)

        return -best_eval

    def _find_best_move(self, game_state, depth, start_time) -> int:
        """
        Finds the best move by evaluating each legal move using the MinMax algorithm to the given depth.
        Time checks are made during the evaluation so that the search can be aborted if necessary.
        """
        best_move = None
        best_value = float('-inf')

        for move in game_state.get_moves():
            if time.time() - start_time >= self.time_limit:
                raise Timeout("Time limit reached during search")
            simulated_game_state = game_state.clone()
            simulated_game_state.apply_move(move)
            move_value = self._min_max(simulated_game_state, depth - 1, start_time)
            if move_value > best_value:
                best_value = move_value
                best_move = move

        return best_move