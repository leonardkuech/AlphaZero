import numpy as np
import pandas as pd
import torch

from CNN import GliderCNN
from Game import Game
from NNetAgent import NNetAgent
from RandomAgent import RandomAgent
from TimedMCTSAgent import TimedMCTSAgent
from Utils import sum_reserve

SIMULATIONS = 500

def run():
    columns = ["Game", "Winner", "Turns", "AZPoints", "RandomPoints"]
    games_df = pd.DataFrame(columns=columns)
    nnet = torch.load('../models/sugar_gliders_nnet1739457037.9798021.pth')
    for i in range(SIMULATIONS):
        if i % 2 == 0:
            policy_agent = NNetAgent(nnet, "AZ")
            random_agent = RandomAgent("RandomAgent")
            game = Game.create_agent_game(policy_agent, random_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"AZPolicy", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]
            else:
                game_stats = [i ,"RandomPolicy", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]

            games_df.loc[i] = game_stats

        else:
            random_agent = RandomAgent("RandomAgent")
            policy_agent = NNetAgent(nnet, "AZ")
            game = Game.create_agent_game(random_agent, policy_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()
            if winner == 0:
                game_stats = [i, "RandomPolicy", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]
            else:
                game_stats = [i, "AZPolicy", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/RandomVSAZ.csv", index=False)

if __name__ == '__main__':
    run()