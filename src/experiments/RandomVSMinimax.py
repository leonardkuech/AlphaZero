import numpy as np
import pandas as pd

from Game import Game
from RandomAgent import RandomAgent
from TimedMinimaxAgent import TimedMinimaxAgent

SIMULATIONS = 100
def run():
    columns = ["Game", "Winner", "Turns"]
    games_df = pd.DataFrame(columns=columns)
    for i in range(SIMULATIONS):
        if i % 2 == 0:
            mcts_agent = TimedMinimaxAgent("MinimaxAgent", 0, 1.0)
            random_agent = RandomAgent("RandomAgent")
            game = Game.create_agent_game(mcts_agent, random_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"MinimaxAgent", game.turns]
            else:
                game_stats = [i ,"RandomPolicy", game.turns]

            games_df.loc[i] = game_stats

        else:
            random_agent = RandomAgent("RandomAgent")
            mcts_agent = TimedMinimaxAgent("MinimaxAgent", 1)
            game = Game.create_agent_game(random_agent, mcts_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()
            if winner == 0:
                game_stats = [i, "RandomPolicy", game.turns]
            else:
                game_stats = [i, "MinimaxAgent", game.turns]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/RandomVSMinimax.csv", index=False)

if __name__ == '__main__':
    run()