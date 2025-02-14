import pandas as pd
import torch
from sympy.strategies.tree import greedy

from GreedyAgent import GreedyAgent
from NNetAgent import NNetAgent
from Game import Game
import logging
from CNN import *
from RandomAgent import RandomAgent
from Utils import sum_reserve

logger = logging.getLogger(__name__)

def main():

    columns = ["Generation", "Winrate"]

    winrate_AZVRandom = pd.DataFrame(columns=columns)

    models = ["sugar_gliders_nnet1739221639.690707.pth", "sugar_gliders_nnet1739222263.707514.pth", "sugar_gliders_nnet1739223981.391359.pth",
              "sugar_gliders_nnet1739232202.4662662.pth", "sugar_gliders_nnet1739234127.2925708.pth", "sugar_gliders_nnet1739241115.46085.pth",
              "sugar_gliders_nnet1739256673.5185611.pth", "sugar_gliders_nnet1739378229.9787898.pth", "sugar_gliders_nnet1739413207.444371.pth",
              "sugar_gliders_nnet1739417305.409232.pth", "sugar_gliders_nnet1739419355.258095.pth", "sugar_gliders_nnet1739421493.341859.pth",
              "sugar_gliders_nnet1739450286.416115.pth", "sugar_gliders_nnet1739452636.402634.pth", "sugar_gliders_nnet1739454827.766761.pth",
              "sugar_gliders_nnet1739457037.9798021.pth"]

    for i in range(len(models)):

        gamesWon = 0
        for j in range(500):

            nnet = torch.load(f'../models/{models[i]}')
            az_agent = NNetAgent(nnet, "NewAgent")
            greedy_agent = GreedyAgent("GreedyAgent")

            if j % 2 == 0:

                game = Game.create_agent_game(az_agent, greedy_agent)
                game.init()
                game.start()
                winner = game.game_state.get_leader()

                if winner == 0:
                    gamesWon += 1

            else:
                game = Game.create_agent_game(greedy_agent, az_agent)

                game.init()
                game.start()

                winner = game.game_state.get_leader()

                if winner == 1:
                    gamesWon += 1

        winrate_AZVRandom.loc[i] = [i , gamesWon/ 500]

    winrate_AZVRandom.to_csv("../data/AZVsGreedyWinrate.csv", index=False)



if __name__ == '__main__':
    main()