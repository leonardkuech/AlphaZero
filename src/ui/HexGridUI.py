import tkinter as tk
import math

import numpy as np

from GameState import GameState
from Utils import inverse_calc_cantor


class Hexagon:
    def __init__(self, tile: int):
        self.c = tile
        x, y = inverse_calc_cantor(tile)
        self.x = int(x * 36 * math.sqrt(3.0) +
                     y * 36 * (math.sqrt(3.0) / 2) + HexGridUI.OFFSET)
        self.y = int(y * 36 * (3.0 / 2.0) + HexGridUI.OFFSET)
        self.color = self.get_color()

    def get_color(self):
        return f"#{100 + (self.x * 10) % 155:02x}{100 + (self.y * 10) % 155:02x}ff"

    def contains(self, point):
        x, y = point
        dx = abs(self.x - x)
        dy = abs(self.y - y)
        return dx ** 2 + dy ** 2 <= HexGridUI.HEX_RADIUS ** 2


class HexGridUI(tk.Canvas):
    HEX_RADIUS = 30
    OFFSET = 400

    def __init__(self, master, game_state: GameState, show_indexes=False, **kwargs):
        super().__init__(master, width=800, height=800, bg="white", **kwargs)

        self.hexagons: list[Hexagon] = []
        self.show_indexes = show_indexes
        self.hovered_hex : Hexagon | None = None
        self.clicked_hex : Hexagon | None= None
        self.game_state : GameState = game_state
        self.pass_next_turn : bool = False
        self.pack()

        self.bind("<Motion>", self.on_mouse_hover)  # Bind for hover effect
        self.bind("<Button-1>", self.on_mouse_click)

        self.initialize_board()

    def initialize_board(self):

        game_board = self.game_state.game_board
        # Generate the hexagon grid
        self.hexagons = [Hexagon(key) for key in game_board.keys()]

        self.update_board()

    def update_board(self):
        self.delete("all")  # Clear the visual representation

        if self.game_state.game_started:
            pass_button = tk.Button(
                self,
                text="Pass",
                command=self.pass_turn,
                font=("Arial", 16, "bold"),
                bg="lightblue",
                fg="white",
                relief="raised",
                bd=3,
                padx=20,
                pady=10
            )
            self.create_window(400, 750, window=pass_button)

        self.draw_all_hexagons()
        self.create_information_panel()

    def draw_all_hexagons(self):
        for hexagon in self.hexagons:
            self.draw_hexagon(hexagon)

    def create_information_panel(self):
        player_info = tk.Frame(self, bg="lightgray", bd=2, relief="solid")
        player_info.place(x=5, y=5, width=795, height=120)

        player_one_reserve = self.game_state.reserves[0]
        player_two_reserve = self.game_state.reserves[1]

        player_one_reserve_sum = sum((i + 1) * player_one_reserve[i] for i in range(len(player_one_reserve)))
        player_two_reserve_sum = sum((i + 1) * player_two_reserve[i] for i in range(len(player_two_reserve)))

        # Create labels with headers
        labels = "1 | 2 | 3 | 4 | 5 | Sum"
        header = tk.Label(player_info, text="Reserve Values", font=("Arial", 16, "bold"), bg="lightgray")
        header.grid(row=0, column=0, columnspan=3, pady=5)

        # Add grid layout for better alignment
        tk.Label(player_info, text=labels, font=("Arial", 14), bg="lightgray").grid(row=1, column=1, sticky="w",
                                                                                    padx=10)

        # Player one info
        tk.Label(player_info, text="Player 1 :", font=("Arial", 14, "bold"), bg="lightgray").grid(row=2, column=0,
                                                                                                  sticky="e", padx=10)
        player_one_label = f"   {player_one_reserve[0]} | {player_one_reserve[1]} | {player_one_reserve[2]} | " \
                           f"{player_one_reserve[3]} | {player_one_reserve[4]} | {player_one_reserve_sum}"
        tk.Label(player_info, text=player_one_label, font=("Arial", 14), bg="lightgray").grid(row=2, column=1,
                                                                                              sticky="w")

        # Player two info
        tk.Label(player_info, text="Player 2 :", font=("Arial", 14, "bold"), bg="lightgray").grid(row=3, column=0,
                                                                                                  sticky="e", padx=10)
        player_two_label = f"   {player_two_reserve[0]} | {player_two_reserve[1]} | {player_two_reserve[2]} | " \
                           f"{player_two_reserve[3]} | {player_two_reserve[4]} | {player_two_reserve_sum}"
        tk.Label(player_info, text=player_two_label, font=("Arial", 14), bg="lightgray").grid(row=3, column=1,
                                                                                              sticky="w")

    def draw_hexagon(self, hexagon: Hexagon):
        points = self.get_hexagon_shape(hexagon.x, hexagon.y)
        fill_color = "yellow" if hexagon == self.clicked_hex else "cyan" if hexagon == self.hovered_hex else hexagon.color
        self.create_polygon(points, outline="white", fill=fill_color, tags="hex")

        if self.game_state.game_board[hexagon.c] > 0:
            self.create_text(hexagon.x, hexagon.y, text=str(self.game_state.game_board[hexagon.c]), font=("Arial", 12, "bold"))

        if self.show_indexes:
            self.create_text(hexagon.x, hexagon.y + 10,
                             text=f'{hexagon.x }|{hexagon.y}',
                             font=("Arial", 10))

        if hexagon.c in self.game_state.positions:
            self.draw_glider(hexagon)

    def draw_glider(self, hexagon):
        indices = np.where(self.game_state.positions == hexagon.c)[0]
        index = indices[0]
        color = "black" if index == 1 else "white" if index == 0 else "red"
        self.create_oval(
            hexagon.x + 7, hexagon.y - 7, hexagon.x + 21, hexagon.y + 7, fill=color, outline=color
        )

    def get_hexagon_shape(self, x, y):
        points = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            px = x + self.HEX_RADIUS * math.cos(angle)
            py = y + self.HEX_RADIUS * math.sin(angle)
            points.append((px, py))
        return points

    def on_mouse_click(self, event):
        point = (event.x, event.y)
        self.clicked_hex = next((hexagon for hexagon in self.hexagons if hexagon.contains(point)), None)

    def on_mouse_hover(self, event):
        point = (event.x, event.y)
        hovered_hex = next((hexagon for hexagon in self.hexagons if hexagon.contains(point)), None)
        if hovered_hex != self.hovered_hex:
            self.hovered_hex = hovered_hex
            self.update_board()  # Redraw the board to update the hover effect

    def pass_turn(self):
        self.pass_next_turn = True

    def get_pass_turn(self):
        passing = self.pass_next_turn
        self.pass_next_turn = False
        return passing

    def get_selected_hex(self):
        if self.clicked_hex is not None:
            clicked_hex_cantor = self.clicked_hex.c
            self.clicked_hex = None
            return clicked_hex_cantor
        else:
            return -1
