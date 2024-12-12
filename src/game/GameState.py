from typing import List

import torch
import matplotlib.pyplot as plt

from Cantor import calc_cantor
from Hextile import HexTile


class GameState:
    def __init__(self, game_board, players: List, player_to_move: int):
        self.players = players
        self.game_board = game_board
        self.player_to_move = player_to_move
        self.game_started = False
        self.not_moved_count = 0
        self.placed = 0

    @staticmethod
    def in_bounds(x: int, y: int, z: int) -> bool:
        return abs(x) < 5 and abs(y) < 5 and abs(z) < 5

    def get_game_board(self):
        return self.game_board

    def set_game_board(self, game_board):
        self.game_board = game_board

    def get_player_to_move(self):
        return self.players[self.player_to_move]

    def get_player_id_to_move(self):
        return self.player_to_move

    def set_player_id_to_move(self, player_to_move: int):
        self.player_to_move = player_to_move

    def get_not_moved_count(self):
        return self.not_moved_count

    def set_not_moved_count(self, not_moved_count: int):
        self.not_moved_count = not_moved_count

    def get_players(self):
        return self.players

    def get_player(self, player_id: int):
        return self.players[player_id]

    def next_player_to_move(self):
        self.player_to_move = (self.player_to_move + 1) % len(self.players)

    def increment_not_moved_count(self):
        self.not_moved_count += 1

    def is_game_started(self) -> bool:
        return self.game_started

    def set_game_started(self, is_started: bool):
        self.game_started = is_started

    def check_game_over(self) -> bool:
        return self.not_moved_count >= 2 or self.not_winnable()

    def not_winnable(self):
        return abs(self.players[0].get_bank() - self.players[1].get_bank()) > self.game_board.get_remaining_tiles_value()

    def check_bank(self) -> List[int]:
        return [player.get_bank() for player in self.players]

    def get_player_on_hex(self, x: int, y: int) -> int:
        for idx, player in enumerate(self.players):
            if player.get_pos() == calc_cantor(x, y):
                return idx
        return -1

    def get_score(self, player: int) -> int:
        tile_value = self.players[player].get_current_hex(self.game_board).get_value()
        return self.players[player].get_bank() + max(tile_value, 0)

    def get_leader(self) -> int:
        scores = [player.get_bank() for player in self.players]
        if scores[0] == scores[1]:
            return -1
        return scores.index(max(scores))

    def apply_move(self, move: int) -> bool:
        if self.game_started:
            if move == float('-inf'):
                self.increment_not_moved_count()
                self.next_player_to_move()
                return True
            elif self.check_move_valid(move):
                self.set_not_moved_count(0)
                current_player = self.get_player_to_move()
                stop = self.game_board.get_tile(c=move)
                start = self.game_board.get_tile(c=current_player.get_pos())

                if start.get_value() > 0:
                    current_player.add_to_reserve(start.get_value())
                    start.set_value(0)
                elif start.get_x() == 0 and start.get_y() == 0 and start.get_z() == 0:
                    if not current_player.subtract_lowest_from_reserve():
                        return False
                else:
                    distance = HexTile.get_distance(start, stop)
                    if not current_player.subtract_from_reserve(distance):
                        return False

                stop.set_occupied(True)
                start.set_occupied(False)
                current_player.set_pos(stop.get_x(), stop.get_y())
                self.next_player_to_move()
                return True
            return False
        else:
            hex_tile = self.game_board.get_tile(c=move)
            if hex_tile.is_occupied or hex_tile.get_value() != 1:
                return False
            hex_tile.set_occupied(True)
            current_player = self.get_player_to_move()
            current_player.set_pos(hex_tile.x, hex_tile.y)
            self.placed += 1
            if self.placed >= 2:
                self.game_started = True
            self.next_player_to_move()
            return True

    def check_move_valid(self, move: int) -> bool:
        return move in self.get_all_possible_moves()

    def get_moves(self) -> List[int]:
        if self.game_started:
            return self.get_all_possible_moves_with_passing()
        return self.game_board.get_placement_tile_ids()

    def get_moves_without_passing(self) -> List[int]:
        if self.game_started:
            return self.get_all_possible_moves()
        return self.game_board.get_placement_tile_ids()

    def get_positive_moves(self) -> List[int]:
        if self.game_started:
            return self.get_all_positive_moves()
        return self.game_board.get_placement_tile_ids()

    def get_all_possible_moves_with_passing(self) -> List[int]:
        return self.get_all_possible_moves() + [float('-inf')]

    def get_all_positive_moves(self):
        player = self.get_player_to_move()
        start = self.game_board.get_tile(player.get_pos_x(), player.get_pos_y())

        moves = []

        if start.get_value() > 0:
            range_value = start.get_value()
            moves = self.get_moves_with_range(range_value, start)
            moves = [move for move in moves if not self.player_is_in_way(start.get_cantor(), move)]
        else:
            if start.get_x() == 0 and start.get_y() == 0 and start.get_z() == 0:
                if not player.reserve_is_empty():
                    for i in range(-4, 5):
                        for j in range(-4, 5):
                            for k in range(-4, 5):
                                if i + j + k == 0:
                                    moves.append(calc_cantor(i, j))
            else:
                reserve = player.get_reserve()
                for i in range(len(reserve)):
                    if reserve[i] > 0:
                        moves.extend(self.get_moves_with_range(i + 1, start))

                moves = [
                    move for move in moves
                    if
                    not self.player_is_in_way(start.get_cantor(), move) and self.expected_move_value(start.get_cantor(),
                                                                                                     move) >= 0
                ]

        return moves

    def get_all_possible_moves(self) -> List[int]:
        # Assuming Player and HexTile have similar methods to Java counterparts
        player = self.get_player_to_move()
        start_tile = self.game_board.get_tile(c=player.get_pos())
        moves = []
        # Logic for computing moves
        if (start_tile.value > 0):
            moves.extend(self.get_moves_with_range(start_tile.value, start_tile))
            moves = [move for move in moves if not self.player_is_in_way(start_tile.get_cantor(), move)]


        elif start_tile.x == 0 and start_tile.y == 0:
            if (not player.reserve_is_empty()):
                for i in range(-4, 5):
                    for j in range(-4, 5):
                        for k in range(-4, 5):
                            if i + j + k == 0 and not self.game_board.get_tile(c=calc_cantor(i, j)).is_occupied:
                                moves.append(calc_cantor(i, j))

        else:
            for i in range(len(player.reserve)):
                if player.reserve[i] > 0:
                    moves.extend(self.get_moves_with_range(i + 1, start_tile))
            moves = [move for move in moves if not self.player_is_in_way(start_tile.get_cantor(), move)]

        return moves

    def get_moves_with_range(self, range_value, start: HexTile):
        moves = []
        if self.in_bounds(start.get_x() + range_value, start.get_y() - range_value, start.get_z()):
            moves.append(calc_cantor(start.get_x() + range_value, start.get_y() - range_value))
        if self.in_bounds(start.get_x() - range_value, start.get_y() + range_value, start.get_z()):
            moves.append(calc_cantor(start.get_x() - range_value, start.get_y() + range_value))
        if self.in_bounds(start.get_x() + range_value, start.get_y(), start.get_z() - range_value):
            moves.append(calc_cantor(start.get_x() + range_value, start.get_y()))
        if self.in_bounds(start.get_x() - range_value, start.get_y(), start.get_z() + range_value):
            moves.append(calc_cantor(start.get_x() - range_value, start.get_y()))
        if self.in_bounds(start.get_x(), start.get_y() + range_value, start.get_z() - range_value):
            moves.append(calc_cantor(start.get_x(), start.get_y() + range_value))
        if self.in_bounds(start.get_x(), start.get_y() - range_value, start.get_z() + range_value):
            moves.append(calc_cantor(start.get_x(), start.get_y() - range_value))
        return moves

    def player_is_in_way(self, start, end):
        start_hex = self.game_board.get_tile(c=start)
        end_hex = self.game_board.get_tile(c=end)
        movement = HexTile.subtract(end_hex, start_hex)
        inc_x = 0 if movement[0] == 0 else movement[0] // abs(movement[0])
        inc_y = 0 if movement[1] == 0 else movement[1] // abs(movement[1])

        x, y = start_hex.get_x(), start_hex.get_y()

        while True:
            x += inc_x
            y += inc_y
            hex_tile = self.game_board.get_tile(x, y)
            if hex_tile.is_occupied:
                return True
            if hex_tile == end_hex:
                break

        return False

    def expected_move_value(self, start, end):
        start_hex = self.game_board.get_tile(c=start)
        end_hex = self.game_board.get_tile(c=end)
        player = self.get_player_to_move()

        if start_hex.get_value() > 0:
            return end_hex.get_value()
        elif start_hex.get_x() == 0 and start_hex.get_y() == 0:
            return end_hex.get_value() - player.get_lowest_from_reserve()
        else:
            return end_hex.get_value() - HexTile.get_distance(start_hex, end_hex)

    def clone(self):
        cloned_board = self.game_board.clone()
        cloned_players = [player.clone() for player in self.players]
        cloned_state = GameState(cloned_board, cloned_players, self.player_to_move)
        cloned_state.set_not_moved_count(self.not_moved_count)
        cloned_state.set_game_started(self.game_started)
        return cloned_state

    def encode(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        grid_size = 9  # 5-hex radius translates to a 11x11 grid
        num_value_channels = 6  # Possible tile values: 0-5
        num_player_channels = 2  # Two players
        num_channels = num_value_channels + num_player_channels

        # Initialize a tensor with zeros
        board_tensor = torch.zeros((num_channels, grid_size, grid_size), dtype=torch.float32)
        player1_points_tensor = torch.zeros(5, dtype=torch.float32)
        player2_points_tensor = torch.zeros(5, dtype=torch.float32)

        for tile in self.game_board.get_all_tiles():
            x, y = tile.x, tile.y
            board_tensor[tile.get_value(), x + 4 , y + 4] = 1.0

            if tile.is_occupied:
                board_tensor[num_value_channels + self.get_player_on_hex(x,y), x + 4, y + 4] = 1.0

        reserve1 = self.players[0].get_reserve()
        for i in range(len(reserve1)):
            player1_points_tensor[i] = reserve1[i]

        reserve2 = self.players[1].get_reserve()
        for i in range(len(reserve2)):
            player1_points_tensor[i] = reserve2[i]

#        for i in range(board_tensor.shape[0]):
#            plt.imshow(board_tensor[i].numpy(), cmap='gray')
#            plt.title(f"Channel {i}")
#            plt.colorbar()
#            plt.show()
