import pandas as pd
import torch

from NNetAgent import NNetAgent
from Game import Game
import logging
from CNN import *
from Utils import sum_reserve

logger = logging.getLogger(__name__)

def main():
    games_won = 0
    player_id_new_agent = 0

    columns = ["Game", "Winner", "Turns", "AZPoints", "RandomPoints"]
    games_df = pd.DataFrame(columns=columns)

    for i in range(1000):
        old_nnet = torch.load("../models/sugar_gliders_nnet1739221639.690707.pth")
        new_nnet = torch.load("../models/sugar_gliders_nnet1739256673.5185611.pth")
        agent_old = NNetAgent(old_nnet, "OldAgent")
        agent_new = NNetAgent(new_nnet, "NewAgent")

        if i % 2 == 0:
            new_policy_agent = NNetAgent(new_nnet, "AZ")
            old_policy_agent = NNetAgent(old_nnet, "AZ")
            game = Game.create_agent_game(old_policy_agent, new_policy_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i, "OldAZPolicy", game.turns, sum_reserve(game.game_state.reserves[0]),
                              sum_reserve(game.game_state.reserves[1])]
            else:
                game_stats = [i, "NewAZPolicy", game.turns, sum_reserve(game.game_state.reserves[0]),
                              sum_reserve(game.game_state.reserves[1])]

            games_df.loc[i] = game_stats

        else:
            new_policy_agent = NNetAgent(new_nnet, "AZ")
            old_policy_agent = NNetAgent(old_nnet, "AZ")
            game = Game.create_agent_game(new_policy_agent, old_policy_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()
            if winner == 0:
                game_stats = [i, "NewAZPolicy", game.turns, sum_reserve(game.game_state.reserves[1]),
                              sum_reserve(game.game_state.reserves[0])]
            else:
                game_stats = [i, "OldAZPolicy", game.turns, sum_reserve(game.game_state.reserves[1]),
                              sum_reserve(game.game_state.reserves[0])]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/Gen1VsGen7.csv", index=False)



if __name__ == '__main__':
    main()