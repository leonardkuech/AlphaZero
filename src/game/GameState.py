import numpy as np

from Utils import calc_cantor, sum_reserve, distance, inverse_calc_cantor, in_bounds, sum_board_values
from numba import int32, boolean, types, njit
from numba.experimental import jitclass
from numba.typed import List, Dict

@njit
def create_game_board() -> dict[int, int]:
    board = {}

    n = 4

    values = np.repeat(np.arange(1, 6, dtype=np.int32), 12)
    values = np.random.permutation(values)

    fruit_index = 0

    for i in range(-n, n + 1):
        r1 = max(-n, -i - n)
        r2 = min(n, -i + n)
        for j in range(r1, r2 + 1):
            if i + j + (-i-j) == 0:
                if j == 0 and i == 0:
                    board[calc_cantor(i,j)] = 0
                else:
                    board[calc_cantor(i,j)] = values[fruit_index]
                    fruit_index += 1
    return board

spec = [
    ('player', int32),
    ('positions', types.Array(int32, 1, 'C')),
    ('reserves', types.Array(int32, 2, 'C')),
    ('game_board', types.DictType(int32, int32)),
    ('player_to_move', int32),
    ('game_started', boolean),
    ('not_moved_count', int32),
    ('placed', int32),
    ('valid_moves', types.ListType(int32)),
    ('valid_moves_calculated', boolean)
]

@jitclass(spec)
class GameState:
    def __init__(self):
        self.player = 2
        # Use a NumPy array with the correct dtype.
        self.positions = np.full(2, -1, dtype=np.int32)
        # Create a 2x5 NumPy array for reserves.
        self.reserves = np.zeros((2, 5), dtype=np.int32)
        # Create a typed dictionary for the game board.
        py_board = create_game_board()
        self.game_board = Dict.empty(key_type=int32, value_type=int32)
        for k, v in py_board.items():
            self.game_board[k] = v

        self.player_to_move = 0
        self.game_started = False
        self.not_moved_count = 0
        self.placed = 0
        # Create an empty typed list for valid_moves.
        self.valid_moves = List.empty_list(int32)
        self.valid_moves_calculated = False

    def next_player_to_move(self):
        self.player_to_move = (self.player_to_move + 1) % self.player

    def increment_not_moved_count(self):
        self.not_moved_count += 1

    def check_game_over(self) -> bool:
        return self.not_moved_count >= 2 or self.not_winnable()

    def not_winnable(self) -> bool:
        return abs(
            sum_reserve(self.reserves[0]) - sum_reserve(self.reserves[1])) > sum_board_values(self.game_board)

    def get_player_on_hex(self, c: int) -> int:
        for i, pos in enumerate(self.positions):
            if pos == c:
                return i

    def get_score(self, player: int) -> int:
        tile_value = self.game_board[self.positions[player]]
        return sum_reserve(self.reserves[player]) + tile_value

    def get_leader(self) -> int:
        scores = [sum_reserve(reserve) for reserve in self.reserves]
        if scores[0] == scores[1]:
            return -1
        return scores.index(max(scores))

    def apply_move(self, move: int) -> bool:
        if self.game_started:
            if move < 0:
                current_player = self.player_to_move
                start = self.positions[current_player]

                if self.game_board[start] > 0:
                    self.reserves[current_player][self.game_board[start] - 1] += 1
                    self.game_board[start] = 0
                self.increment_not_moved_count()
                self.next_player_to_move()
                self.valid_moves_calculated = False
                return True
            elif self.check_move_valid(move):
                self.not_moved_count = 0
                current_player = self.player_to_move
                start = self.positions[current_player].item()

                if self.game_board[start] > 0:
                    self.reserves[current_player][self.game_board[start] - 1] += 1
                    self.game_board[start] = 0

                else:

                    x, y = inverse_calc_cantor(start)
                    if x == 0 and y == 0:

                        lowest_index = -1
                        index = 0
                        for x in self.reserves[current_player]:
                            if x > 0:
                                lowest_index = index
                                break
                            index += 1
                        if lowest_index >= 0:
                            self.reserves[current_player][lowest_index] -= 1
                        else:
                            return False
                    else:
                        if self.reserves[current_player][distance(start, move) - 1] > 0:
                            self.reserves[current_player][distance(start, move) - 1] -= 1
                        else:
                            return False

                self.positions[current_player] = move
                self.next_player_to_move()
                self.valid_moves_calculated = False
                return True
            return False
        else:
            if self.check_move_valid(move):
                current_player = self.player_to_move
                self.positions[current_player] = move
                self.placed += 1
                if self.placed >= 2:
                    self.game_started = True
                self.next_player_to_move()
                self.valid_moves_calculated = False
                return True
            return False


    def check_move_valid(self, move: int) -> bool:
        if self.game_started:
            if move < 0:
                return True
            return move in self.get_all_possible_moves()
        else:
            if move < 0:
                return False
            return move not in self.positions and self.game_board[move] == 1

    def get_placement_tiles(self) -> list[int]:
        placement_tiles = List.empty_list(int32)
        for key in self.game_board:
            if self.game_board[key] == 1 and key not in self.positions:
                placement_tiles.append(key)
        return placement_tiles

    def get_moves(self) -> list[int]:
        if self.game_started:
            return self.get_all_possible_moves_with_passing()
        return self.get_placement_tiles()

    def get_moves_without_passing(self) -> list[int]:
        if self.game_started:
            return self.get_all_possible_moves()
        return self.get_placement_tiles()

    def get_all_possible_moves_with_passing(self) -> list[int]:
        moves = self.get_all_possible_moves()
        moves = moves.copy()
        moves.append(np.int32(-1))
        return moves

    def get_all_possible_moves(self) -> list[int]:
        if self.valid_moves_calculated:
            return self.valid_moves

        player = self.player_to_move
        start = self.positions[player].item()

        moves = List.empty_list(int32)

        if self.game_board[start] > 0:
            range_value = self.game_board[start]
            moves = self.get_moves_with_range(range_value, start)
            moves = self.remove_not_reachable(start, moves)

        else:

            x, y = inverse_calc_cantor(start)

            if x == 0 and y == 0:

                if not sum_reserve(self.reserves[player]) == 0:

                    for i in range(-4, 5):

                        for j in range(-4, 5):

                            for k in range(-4, 5):

                                if i + j + k == 0:
                                    moves.append(calc_cantor(i, j))
            else:
                for i in range(len(self.reserves[player])):
                    if self.reserves[player][i] > 0:
                        moves.extend(self.get_moves_with_range(i + 1, start))
                moves = self.remove_not_reachable(start, moves)

        self.valid_moves = moves
        self.valid_moves_calculated = True

        return self.valid_moves

    def get_moves_with_range(self, range_value, start: int):
        moves = List.empty_list(int32)
        x, y = inverse_calc_cantor(start)
        z = - x - y

        if in_bounds(x + range_value, y - range_value, z):
            moves.append(calc_cantor(x + range_value, y - range_value))
        if in_bounds(x - range_value, y + range_value, z):
            moves.append(calc_cantor(x - range_value, y + range_value))
        if in_bounds(x + range_value, y, z - range_value):
            moves.append(calc_cantor(x + range_value, y))
        if in_bounds(x - range_value, y, z + range_value):
            moves.append(calc_cantor(x - range_value, y))
        if in_bounds(x, y + range_value, z - range_value):
            moves.append(calc_cantor(x, y + range_value))
        if in_bounds(x, y - range_value, z + range_value):
            moves.append(calc_cantor(x, y - range_value))
        return moves

    def remove_not_reachable(self, start : int, moves: List[int32]) -> List[int32]:
        filtered_moves = List.empty_list(int32)
        for move in moves:
            if not self.player_is_in_way(start, move):
                filtered_moves.append(move)
        return filtered_moves

    def player_is_in_way(self, start, end):
        x, y = inverse_calc_cantor(start)
        xe, ye = inverse_calc_cantor(end)


        movement = [xe - x, ye - y]

        inc_x = 0 if movement[0] == 0 else movement[0] // abs(movement[0])
        inc_y = 0 if movement[1] == 0 else movement[1] // abs(movement[1])

        while True:
            x += inc_x
            y += inc_y

            c = calc_cantor(x, y)

            if c in self.positions:
                return True
            if c == end:
                break

        return False

    def clone(self) -> "GameState":
        # Create a new instance of GameState
        cloned = GameState()

        cloned.player = self.player
        cloned.player_to_move = self.player_to_move
        cloned.game_started = self.game_started
        cloned.not_moved_count = self.not_moved_count
        cloned.placed = self.placed
        cloned.valid_moves_calculated = self.valid_moves_calculated

        cloned.positions = self.positions.copy()
        cloned.reserves = self.reserves.copy()

        cloned.game_board = Dict.empty(key_type=int32, value_type=int32)
        for k in self.game_board:
            cloned.game_board[k] = self.game_board[k]

        cloned.valid_moves = List.empty_list(int32)
        for mv in self.valid_moves:
            cloned.valid_moves.append(mv)

        return cloned

    def encode(self) -> (np.array, np.array, np.array):
        grid_size = 9  # 5-hex radius translates to a 9x9 grid
        num_value_channels = 6  # Possible tile values: 0-5
        num_player_channels = 2  # Two players
        num_channels = num_value_channels + num_player_channels

        # Initialize a tensor with zeros
        board_tensor = np.zeros((1, num_channels, grid_size, grid_size), dtype=np.float32)
        player1_points_tensor = np.zeros((1,5), dtype=np.float32)
        player2_points_tensor = np.zeros((1,5), dtype=np.float32)

        for tile in self.game_board.keys():
            x, y = inverse_calc_cantor(tile)
            board_tensor[0, self.game_board[tile].item(), x + 4, y + 4] = 1.0

        for index, pos in enumerate(self.positions):
            if pos == -1:
                continue

            x, y = inverse_calc_cantor(pos)
            if self.player_to_move == index:
                board_tensor[0, num_value_channels, x + 4, y + 4] = 1.0
            else:
                board_tensor[0, num_value_channels + 1, x + 4, y + 4] = 1.0

        reserve1 = self.reserves[self.player_to_move]

        for i in range(len(reserve1)):
            player1_points_tensor[0,i] = reserve1[i].item()

        reserve2 = self.reserves[self.player_to_move ^ 1]

        for i in range(len(reserve2)):
            player2_points_tensor[0,i] = reserve2[i].item()
        #
        # for i in range(board_tensor.shape[0]):
        #     plt.imshow(board_tensor[i].numpy(), cmap='gray')
        #     plt.title(f"Channel {i}")
        #     plt.colorbar()
        #     plt.show()

        return board_tensor, player1_points_tensor, player2_points_tensor

    def string_representation(self) -> str:
        rep = ''

        for tile in self.game_board.keys():
            rep += str(self.game_board[tile])

        for pos in self.positions:
            rep += str(pos)

        for reserve in self.reserves:
            for x in reserve:
                rep += str(x)

        rep += str(self.player_to_move)

        return rep
