import numpy as np
import pandas as pd
import torch

from CNN import GliderCNN
from Game import Game
from GreedyAgent import GreedyAgent
from MCTSNNetAgent import MCTSNNetAgent
from NNetAgent import NNetAgent
from RandomAgent import RandomAgent
from TimedMCTSAgent import TimedMCTSAgent
from Utils import sum_reserve

SIMULATIONS = 100

def run():
    columns = ["Game", "Winner", "Turns", "AZPoints", "RandomPoints"]
    games_df = pd.DataFrame(columns=columns)
    nnet = torch.load('../models/sugar_gliders_nnet1740344851.2018251.pth')
    for i in range(SIMULATIONS):
        print("Simulation #", i+1)
        if i % 2 == 0:
            policy_agent = NNetAgent(nnet, "AZ")
            random_agent = GreedyAgent("RandomAgent")
            game = Game.create_agent_game(policy_agent, random_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"AZPolicy", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]
            elif winner == 1:
                game_stats = [i, "RandomPolicy", game.turns, sum_reserve(game.game_state.reserves[0]),
                              sum_reserve(game.game_state.reserves[1])]
            else:
                game_stats = [i ,"Draw", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]

            print(game_stats[1])
            games_df.loc[i] = game_stats

        else:
            random_agent = GreedyAgent("RandomAgent")
            policy_agent = NNetAgent(nnet, "AZ")
            game = Game.create_agent_game(random_agent, policy_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()
            if winner == 0:
                game_stats = [i, "RandomPolicy", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]
            elif winner == 1:
                game_stats = [i, "AZPolicy", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]
            else:
                game_stats = [i ,"Draw", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]

            print(game_stats[1])
            games_df.loc[i] = game_stats

    print(games_df["Winner"].value_counts() / SIMULATIONS)
    #games_df.to_csv("../data/RandomVSAZ.csv", index=False)

if __name__ == '__main__':
    run()