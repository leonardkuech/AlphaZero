import pandas as pd
import torch

from Game import Game
from TimedMinimaxAgentPruned import TimedMinimaxAgentPruned
from TimedMinimaxAgentPrunedAZEval import TimedMinimaxAgentPrunedAZ
from Utils import sum_reserve

SIMULATIONS = 500
def run():
    columns = ["Game", "Winner", "Turns", "VanillaPoints", "PrunedPoints"]
    games_df = pd.DataFrame(columns=columns)

    nnet = torch.load('../models/sugar_gliders_nnet1739378229.9787898.pth')

    for i in range(SIMULATIONS):
        if i % 2 == 0:
            pruned_agent = TimedMinimaxAgentPruned("TimedMinimaxAgentPruned",  1.0)
            pruned_agent_az = TimedMinimaxAgentPrunedAZ("TimedMinimaxAgentPruned",nnet, 1.0)
            game = Game.create_agent_game( pruned_agent,  pruned_agent_az)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"PrunedAgent", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]
            else:
                game_stats = [i ,"PrunedAgentAZ", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]

            games_df.loc[i] = game_stats

        else:
            pruned_agent_az = TimedMinimaxAgentPrunedAZ("TimedMinimaxAgentPruned", nnet, 1.0)
            pruned_agent = TimedMinimaxAgentPruned("TimedMinimaxAgentPruned",  1.0)
            game = Game.create_agent_game(pruned_agent_az, pruned_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i, "PrunedAgentAZ", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]
            else:
                game_stats = [i, "PrunedAgent", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/MinimaxPrunedVsMinimaxPrunedAZEval.csv", index=False)

if __name__ == '__main__':
    run()