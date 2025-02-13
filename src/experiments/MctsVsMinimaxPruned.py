import numpy as np
import pandas as pd

from Game import Game
from TimedMCTSAgent import TimedMCTSAgent
from TimedMinimaxAgentPruned import TimedMinimaxAgentPruned
from Utils import sum_reserve

SIMULATIONS = 500
def run():
    columns = ["Game", "Winner", "Turns", "MCTSPoints", "MinimaxPrunedPoints"]
    games_df = pd.DataFrame(columns=columns)
    point_history = []

    for i in range(SIMULATIONS):
        if i % 2 == 0:
            mcts_agent = TimedMCTSAgent("MCTSAgent",  0, 2.0)
            pruned_agent = TimedMinimaxAgentPruned("TimedMinimaxAgentPruned",  2.0)
            game = Game.create_agent_game(mcts_agent, pruned_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"MCTS", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]
            else:
                game_stats = [i ,"PrunedAgent", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]

            games_df.loc[i] = game_stats
            point_history.append(np.array(game.points_history))

        else:
            pruned_agent = TimedMinimaxAgentPruned("TimedMinimaxAgentPruned",  2.0)
            mcts_agent = TimedMCTSAgent("MCTSAgent",  1, 2.0)
            game = Game.create_agent_game(pruned_agent, mcts_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i, "PrunedAgent", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]
            else:
                game_stats = [i, "MCTS", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]

            games_df.loc[i] = game_stats

            point_difference = np.array(game.points_history)
            point_history.append(-np.array(game.points_history))

    games_df.to_csv("../data/MctsVsMinimax2SecPruned.csv", index=False)
    point_history_df = pd.DataFrame(point_history)
    point_history_df.to_csv("../data/MctsVsMinimax2SecPointDifferenceHistory.csv", index=False, header=False)

if __name__ == '__main__':
    run()