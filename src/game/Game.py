import time
import logging

from Agent import Agent
from Player import Player
from GameState import GameState
from Board import Board


logger = logging.getLogger(__name__)

class Game:
    def __init__(self, game_state: GameState):
        self.turns = 2
        self.game_state = game_state
        self.ui = None
        self.has_ui = False


    @staticmethod
    def create_game():
        board = Board.create_new_board()
        player1 = Player(0, "Player One")
        player2 = Player(1, "Player Two")

        game_state = GameState(board, [player1, player2], 0)
        return Game(game_state)

    @staticmethod
    def create_game_with_agent(agent: Agent):
        board = Board.create_new_board()
        player1 = Player(1, "Human Player")
        player2 = Player(2, agent=agent)

        game_state = GameState(board, [player1, player2], 0)
        return Game(game_state)

    @staticmethod
    def create_agent_game(agent1: Agent, agent2: Agent):
        board = Board.create_new_board()
        player1 = Player(0, agent=agent1)
        player2 = Player(1, agent=agent2)

        game_state = GameState(board, [player1, player2], 0)
        return Game(game_state)

    def get_game_state(self):
        return self.game_state

    def make_move(self, move: int) -> bool:
        return self.game_state.apply_move(move)

    def init(self):
        for player in self.game_state.get_players():
            if player.is_agent:
                while True:
                    selected = player.choose_move(self.game_state)
                    if selected != -1 and self.make_move(selected):
                        break
                    # time.sleep(2)
            else:
                while True:
                    selected = self.ui.get_selected_hex()
                    if selected != -1 and self.make_move(selected):
                        break
                    # time.sleep(0.2)

            if self.has_ui:
                self.ui.update_board()

            # time.sleep(0.2)

        if self.has_ui:
            self.ui.update_board()

        # time.sleep(1)

    def start(self):
        while self.game_state.is_game_started():
            player = self.game_state.get_player_to_move()
            if player.is_agent:
                while True:
                    selected = player.choose_move(self.game_state)
                    if self.make_move(selected):
                        if self.has_ui:
                            self.ui.update_board()
                        break
                    # time.sleep(2)
            else:
                while True:
                    if self.ui and self.ui.get_pass_turn() and self.make_move(float('-inf')):
                        if self.has_ui:
                            self.ui.update_board()
                        break

                    selected = self.ui.get_selected_hex()
                    if selected >= 0 and self.make_move(selected):
                        if self.has_ui:
                            self.ui.update_board()
                        # time.sleep(0.2)
                        break
                    # time.sleep(0.2)
            self.turns += 1
            if self.game_state.check_game_over():
                self.game_over()

    def game_over(self):
        print("Game Over")
        logger.info(f'Game Over in {self.turns} turns')
        self.game_state.set_game_started(False)

    def get_ui(self):
        return self.ui

    def set_ui(self, ui):
        self.ui = ui
        self.has_ui = True
