import numpy as np
import pandas as pd

from Game import Game
from GreedyAgent import GreedyAgent
from RandomAgent import RandomAgent
from TimedMCTSAgent import TimedMCTSAgent
from Utils import sum_reserve

SIMULATIONS = 100
def run():
    columns = ["Game", "Winner", "Turns", "MCTSPoints", "GreedyPoints"]
    games_df = pd.DataFrame(columns=columns)
    for i in range(SIMULATIONS):
        if i % 2 == 0:
            mcts_agent = TimedMCTSAgent("MCTSAgent", 0, 1.0)
            greedy_agent = GreedyAgent("GreedyAgent")
            game = Game.create_agent_game(mcts_agent, greedy_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"MCTS", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]
            else:
                game_stats = [i ,"RandomPolicy", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]

            games_df.loc[i] = game_stats

        else:
            greedy_agent = GreedyAgent("GreedyAgent")
            mcts_agent = TimedMCTSAgent("MCTSAgent", 1, 1.0)
            game = Game.create_agent_game(greedy_agent, mcts_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()
            if winner == 0:
                game_stats = [i, "GreedyPolicy", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]
            else:
                game_stats = [i, "MCTS", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/GreedyVSMCTS.csv", index=False)

if __name__ == '__main__':
    run()