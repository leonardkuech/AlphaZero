import numpy as np
import pandas as pd

from Game import Game
from TimedMinimaxAgent import TimedMinimaxAgent
from TimedMCTSAgent import TimedMCTSAgent
from MCTSAgent import MCTSAgent

SIMULATIONS=500
def run():
    columns = ["Game", "Winner", "Turns"]
    games_df = pd.DataFrame(columns=columns)
    for i in range(SIMULATIONS):
        if i % 2 == 0:
            minimax_agent = TimedMinimaxAgent("MinimaxAgent",  1.0)
            mcts_agent = TimedMCTSAgent("TimedMCTSAgent", 1, 1.0)
            game = Game.create_agent_game(minimax_agent, mcts_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"MinimaxAgent", game.turns]
            else:
                game_stats = [i ,"MCTS", game.turns]

            games_df.loc[i] = game_stats

        else:
            mcts_agent = TimedMCTSAgent("TimedMCTSAgent", 0, 1.0)
            minimax_agent = TimedMinimaxAgent("MinimaxAgent",  1.0)
            game = Game.create_agent_game(mcts_agent, minimax_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i, "MCTS", game.turns]
            else:
                game_stats = [i, "MinimaxAgent", game.turns]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/MinimaxVSMcts.csv", index=False)

if __name__ == '__main__':
    run()