import pandas as pd

from Game import Game
from TimedMCTSAgent import TimedMCTSAgent
from TimedMCTSAgentHardPlayout import TimedMCTSAgentHardPlayout
from Utils import sum_reserve

SIMULATIONS = 100
def run():
    columns = ["Game", "Winner", "Turns", "MCTSPoints", "MCTSHardPoints", "AvgSimulations", "AvgSimulationsHard"]
    games_df = pd.DataFrame(columns=columns)
    for i in range(SIMULATIONS):
        if i % 2 == 0:
            mcts_agent = TimedMCTSAgent("MinimaxAgent",  0, 1.0)
            hard_agent = TimedMCTSAgentHardPlayout("TimedMCTSAgentHardPlayout",  1, 1.0)
            game = Game.create_agent_game(mcts_agent, hard_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"SoftPlayoutAgent", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1]), sum(mcts_agent.simulations) / len(mcts_agent.simulations), sum(hard_agent.simulations) / len(hard_agent.simulations)]
            else:
                game_stats = [i ,"HardPlayoutAgent", game.turns, sum_reserve(game.game_state.reserves[0]), sum_reserve(game.game_state.reserves[1]),  sum(mcts_agent.simulations) / len(mcts_agent.simulations), sum(hard_agent.simulations) / len(hard_agent.simulations)]

            games_df.loc[i] = game_stats

        else:
            hard_agent = TimedMCTSAgentHardPlayout("TimedMCTSAgentHardPlayout", 0, 1.0)
            mcts_agent = TimedMCTSAgent("MinimaxAgent",  1, 1.0)
            game = Game.create_agent_game(hard_agent, mcts_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i, "HardPlayoutAgent", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0]),  sum(mcts_agent.simulations) / len(mcts_agent.simulations), sum(hard_agent.simulations) / len(hard_agent.simulations)]
            else:
                game_stats = [i, "SoftPlayoutAgent", game.turns, sum_reserve(game.game_state.reserves[1]), sum_reserve(game.game_state.reserves[0]),  sum(mcts_agent.simulations) / len(mcts_agent.simulations), sum(hard_agent.simulations) / len(hard_agent.simulations)]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/MctsVsMctsHardPlayout.csv", index=False)

if __name__ == '__main__':
    run()