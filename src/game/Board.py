import random
from typing import Any

from Hextile import HexTile
from Cantor import calc_cantor

class Board:
    def __init__(self):
        self.game_board = {}

    @staticmethod
    def create_new_board():
        board = Board()

        n = 4

        values = Board._create_values()
        fruit_index = 0

        for i in range(-n, n + 1):
            r1 = max(-n, -i - n)
            r2 = min(n, -i + n)
            for j in range(r1, r2 + 1):
                if j == 0 and i == 0:
                    hex_tile = HexTile(i, j, -i - j)
                else:
                    hex_tile = HexTile(i, j, -i - j, values[fruit_index])
                    fruit_index += 1

                if hex_tile.is_valid():
                    board.add_tile(hex_tile)

        return board

    @staticmethod
    def _create_values():
        values = [i for i in range(1, 6) for _ in range(12)]
        random.seed(42)
        random.shuffle(values)
        return values

    def get_game_board(self):
        return self.game_board

    def set_game_board(self, game_board):
        self.game_board = game_board

    def add_tile(self, hex_tile):
        key = calc_cantor(hex_tile.get_x(), hex_tile.get_y())
        self.game_board[key] = hex_tile

    def print_board(self):
        for key, hex_tile in self.game_board.items():
            print(f"x: {hex_tile.get_x()} | y: {hex_tile.get_y()} | z: {hex_tile.get_z()} | value: {hex_tile.get_value()}")

    def get_tile_count(self):
        return len(self.game_board)

    def get_all_tiles(self):
        return list(self.game_board.values())

    def get_placement_tiles(self):
        return [tile for tile in self.game_board.values() if tile.get_value() == 1 and not tile.is_occupied]

    def get_placement_tile_ids(self):
        return [tile.get_cantor() for tile in self.get_placement_tiles()]

    def get_tile(self, x=None, y=None, c=None) -> HexTile | None:
        if c is not None:
            return self.game_board.get(c)
        if x is not None and y is not None:
            return self.game_board.get(calc_cantor(x, y))
        return None

    def clone(self):
        cloned_board = Board()
        cloned_game_board = {key: tile.clone() for key, tile in self.game_board.items()}
        cloned_board.set_game_board(cloned_game_board)
        return cloned_board
