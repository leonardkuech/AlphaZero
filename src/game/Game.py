import logging
import time
import tkinter

from Agent import Agent
from HexGridUI import HexGridUI
from GameState import GameState
from Utils import sum_reserve

logger = logging.getLogger(__name__)

class Game:
    def __init__(self, game_state: GameState, agent1: Agent = None, agent2: Agent = None):
        self.turns : int = 2
        self.points_history = []
        self.game_state : GameState = game_state
        self.ui : HexGridUI | None = None
        self.agents : list[Agent|None] = [agent1, agent2]
        self.has_ui : bool = False


    @staticmethod
    def create_game():

        game_state = GameState()
        return Game(game_state)

    @staticmethod
    def create_game_with_agent(agent: Agent):

        game_state = GameState()
        return Game(game_state, agent2=agent)

    @staticmethod
    def create_agent_game(agent1: Agent, agent2: Agent):

        game_state = GameState()
        return Game(game_state, agent1=agent1, agent2=agent2)

    def make_move(self, move: int | float) -> bool:
        return self.game_state.apply_move(move)

    def init(self):
        for player in range(self.game_state.player):
            if self.agents[player] is not None:
                while True:
                    selected = self.agents[player].choose_move(self.game_state)
                    if selected != -1 and self.make_move(selected):
                        break
            else:
                while True:
                    time.sleep(0.2)
                    selected = self.ui.get_selected_hex()
                    if selected != -1 and self.make_move(selected):
                        break
            if self.has_ui:
                self.ui.update_board()

        if self.has_ui:
            self.ui.update_board()


    def start(self):
        while self.game_state.game_started:
            begin = time.time()
            player = self.game_state.player_to_move
            if self.agents[player] is not None:
                while True:
                    selected = self.agents[player].choose_move(self.game_state)
                    if self.make_move(selected):
                        if self.has_ui:
                            self.ui.update_board()
                        break
                    # time.sleep(2)
            else:
                while True:
                    if self.ui and self.ui.get_pass_turn() and self.make_move(-1):
                        if self.has_ui:
                            self.ui.update_board()
                        break

                    time.sleep(0.2)
                    selected = self.ui.get_selected_hex()
                    if selected >= 0 and self.make_move(selected):
                        if self.has_ui:
                            self.ui.update_board()
                        break
            # print(time.time() - begin)
            self.turns += 1
            if self.turns % 2 == 0:
                self.points_history.append(sum_reserve(self.game_state.reserves[0]) - sum_reserve(self.game_state.reserves[1]))
            if self.game_state.check_game_over():
                self.game_over()

    def game_over(self):
        logger.info('Game Over')
        self.game_state.game_started = False

    def init_ui(self):
        root = tkinter.Tk()
        root.resizable(True, True)
        self.ui = HexGridUI(root, self.game_state, show_indexes=False)
        self.has_ui = True

        return root
