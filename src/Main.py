import threading
import tkinter

from Game import Game
from HexGridUI import HexGridUI
from MCTSAgent import MCTSAgent
from HeuristikMCTSAgent import HeuristicMCTSAgent
from MinMaxAgent import MinMaxAgent
from PrunedMinMaxAgent import PrunedMinMaxAgent
from RandomAgent import RandomAgent
from RandomPositiveAgent import RandomPositiveAgent


def start_game_logic(game, finished_event):
    game.init()
    game.start()
    # Signal that the game thread has finished
    finished_event.set()


def main():
    # Create a new Game
    agent2 = RandomPositiveAgent("RandomAgent")
    # agent2 = PrunedMinMaxAgent(8,1,"MinMaxAgent")
    agent1 = MCTSAgent("MCTSAgent", 0)
    # agent1 = HeuristicMCTSAgent("MCTSAgent", 1)
    # game = Game.create_game_with_agent(agent=agent2)
    game = Game.create_agent_game(agent1, agent2)

    # Initialize the UI in the main thread
    root = tkinter.Tk()
    root.resizable(False, False)
    ui = HexGridUI(root, game, show_indexes=True)
    game.set_ui(ui)

    # Create an event to signal when the game thread is finished
    finished_event = threading.Event()

    # Start the game logic in a separate thread
    game_thread = threading.Thread(target=start_game_logic, args=(game, finished_event))
    game_thread.start()

    # Periodically check if the game thread has finished
    def check_game_thread():
        if finished_event.is_set():
            print("Game finished")
            # root.destroy()  # Close the UI
        else:
            root.after(100, check_game_thread)

    # Start the tkinter main loop (must run in the main thread)
    root.after(100, check_game_thread)
    root.mainloop()



if __name__ == "__main__":
    main()
