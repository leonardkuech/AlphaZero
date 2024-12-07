import tkinter as tk
import math

from Cantor import calc_cantor
from Hextile import HexTile


class HexGridUI(tk.Canvas):
    HEX_RADIUS = 30
    OFFSET = 400

    def __init__(self, master, game, **kwargs):
        super().__init__(master, width=800, height=800, bg="white", **kwargs)
        self.hexagons = []  # Persistent list of hexagons
        self.hovered_hex = None
        self.clicked_hex = None
        self.game = game
        self.pass_next_turn = False
        self.pack()

        self.bind("<Motion>", self.on_mouse_hover)  # Bind for hover effect
        self.bind("<Button-1>", self.on_mouse_click)

        self.initialize_board()

    def initialize_board(self):
        """Initialize the hexagons list without clearing it."""
        print("Initializing board...")
        game_board = self.game.get_game_state().get_game_board()

        # Generate the hexagon grid
        self.hexagons = [self.Hexagon(hex_tile, self) for hex_tile in game_board.get_all_tiles()]

        self.update_board()

    def update_board(self):
        self.delete("all")  # Clear the visual representation

        if self.game.get_game_state().is_game_started():
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

        players = self.game.get_game_state().get_players()
        player_one_reserve = players[0].get_reserve()
        player_two_reserve = players[1].get_reserve()

        player_one_reserve_sum = sum((i + 1) * player_one_reserve[i] for i in range(len(player_one_reserve)))
        player_two_reserve_sum = sum((i + 1) * player_two_reserve[i] for i in range(len(player_two_reserve)))

        # Create labels with headers
        labels = "1 | 2 | 3 | 4 | 5 | Sum"
        header = tk.Label(player_info, text="Reserve Values", font=("Arial", 16, "bold"), bg="lightgray")
        header.grid(row=0, column=0, columnspan=3, pady=5)

        # Add grid layout for better alignment
        tk.Label(player_info, text=labels, font=("Arial", 14), bg="lightgray").grid(row=1, column=1, sticky="w", padx=10)

        # Player one info
        tk.Label(player_info, text="Player 1 :", font=("Arial", 14, "bold"), bg="lightgray").grid(row=2, column=0, sticky="e", padx=10)
        player_one_label = f"   {player_one_reserve[0]} | {player_one_reserve[1]} | {player_one_reserve[2]} | " \
                           f"{player_one_reserve[3]} | {player_one_reserve[4]} | {player_one_reserve_sum}"
        tk.Label(player_info, text=player_one_label, font=("Arial", 14), bg="lightgray").grid(row=2, column=1, sticky="w")

        # Player two info
        tk.Label(player_info, text="Player 2 :", font=("Arial", 14, "bold"), bg="lightgray").grid(row=3, column=0, sticky="e", padx=10)
        player_two_label = f"   {player_two_reserve[0]} | {player_two_reserve[1]} | {player_two_reserve[2]} | " \
                           f"{player_two_reserve[3]} | {player_two_reserve[4]} | {player_two_reserve_sum}"
        tk.Label(player_info, text=player_two_label, font=("Arial", 14), bg="lightgray").grid(row=3, column=1, sticky="w")


    def draw_hexagon(self, hexagon):
        points = self.get_hexagon_shape(hexagon.x, hexagon.y)
        fill_color = "yellow" if hexagon == self.clicked_hex else "cyan" if hexagon == self.hovered_hex else hexagon.color
        self.create_polygon(points, outline="white", fill=fill_color, tags="hex")

        if hexagon.hex_tile.value > 0:
            self.create_text(hexagon.x, hexagon.y, text=str(hexagon.value), font=("Arial", 12, "bold"))

        if hexagon.hex_tile.is_occupied:
            self.draw_glider(hexagon)

    def draw_glider(self, hexagon):
        player = self.game.get_game_state().get_player_on_hex(hexagon.hex_tile.get_x(), hexagon.hex_tile.get_y())
        color = "black" if player == 1 else "white" if player == 0 else "red"
        self.create_oval(
            hexagon.x + 7 , hexagon.y - 7, hexagon.x + 21, hexagon.y + 7, fill=color, outline=color
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
        passing =  self.pass_next_turn
        self.pass_next_turn = False
        return passing

    def get_selected_hex(self):
        if self.clicked_hex is not None:
            clicked_hex_cantor = self.clicked_hex.hex_tile.get_cantor()
            self.clicked_hex = None
            return clicked_hex_cantor
        else:
            return -1

    class Hexagon:
        def __init__(self, hex_tile: HexTile, parent):
            self.parent = parent
            self.hex_tile = hex_tile
            self.x = int(hex_tile.get_x() * 36 * math.sqrt(3.0) +
                         hex_tile.get_y() * 36 * (math.sqrt(3.0) / 2) + HexGridUI.OFFSET)
            self.y = int(hex_tile.get_y() * 36 * (3.0 / 2.0) + HexGridUI.OFFSET)
            self.value = hex_tile.get_value()
            self.color = self.get_color()

        def get_color(self):
            return f"#{100 + (self.x * 10) % 155:02x}{100 + (self.y * 10) % 155:02x}ff"

        def contains(self, point):
            x, y = point
            dx = abs(self.x - x)
            dy = abs(self.y - y)
            return dx ** 2 + dy ** 2 <= HexGridUI.HEX_RADIUS ** 2
