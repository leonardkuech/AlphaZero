import threading
import tkinter

import random

from Game import Game
from HexGridUI import HexGridUI
from MCTSAgent import MCTSAgent
from MinMaxAgent import MinMaxAgent


def start_game_logic(game, finished_event):
    game.init()
    game.start()
    finished_event.set()


def main():
    agent = MCTSAgent('mcts',1)
    #agent = MinMaxAgent('minimax', 2)
    game = Game.create_game_with_agent(agent)

    root = game.init_ui()

    finished_event = threading.Event()

    game_thread = threading.Thread(target=start_game_logic, args=(game, finished_event))
    game_thread.start()

    def check_game_thread():
        if finished_event.is_set():
            print("Game finished")
        else:
            root.after(100, check_game_thread)

    root.after(100, check_game_thread)
    root.mainloop()

if __name__ == "__main__":
    main()
