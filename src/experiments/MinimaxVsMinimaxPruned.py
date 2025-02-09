import pandas as pd

from Game import Game
from TimedMinimaxAgent import TimedMinimaxAgent
from TimedMinimaxAgentPruned import TimedMinimaxAgentPruned
from Utils import sum_reserve

SIMULATIONS = 100
def run():
    columns = ["Game", "Winner", "Turns", "VanillaPoints", "PrunedPoints"]
    games_df = pd.DataFrame(columns=columns)

    for i in range(SIMULATIONS):
        if i % 2 == 0:
            minimax_agent = TimedMinimaxAgent("MinimaxAgent",  1.0)
            pruned_agent = TimedMinimaxAgentPruned("TimedMinimaxAgentPruned",  1.0)
            game = Game.create_agent_game(minimax_agent, pruned_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"MinimaxAgent", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]
            else:
                game_stats = [i ,"PrunedAgent", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1])]

            games_df.loc[i] = game_stats

        else:
            pruned_agent = TimedMinimaxAgentPruned("TimedMinimaxAgentPruned",  1.0)
            minimax_agent = TimedMinimaxAgent("MinimaxAgent",  1.0)
            game = Game.create_agent_game(pruned_agent, minimax_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i, "PrunedAgent", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]
            else:
                game_stats = [i, "MinimaxAgent", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0])]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/MinimaxVSMinimaxPruned.csv", index=False)

if __name__ == '__main__':
    run()