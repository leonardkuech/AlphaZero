import pandas as pd

from Game import Game
from MCTSAgent import MCTSAgent
from MCTSHardPlayout import MCTSAgentHardPlayout

SIMULATIONS = 10
def run():
    columns = ["Game", "Winner", "Turns"]
    games_df = pd.DataFrame(columns=columns)
    for i in range(SIMULATIONS):
        if i % 2 == 0:
            mcts_agent = MCTSAgent("MCTS",  0)
            hard_agent = MCTSAgentHardPlayout("MCTSAgentHardPlayout",  1)
            game = Game.create_agent_game(mcts_agent, hard_agent)
            game.init()
            game.start()
            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i ,"SoftPlayoutAgent", game.turns]
            else:
                game_stats = [i ,"HardPlayoutAgent", game.turns]

            games_df.loc[i] = game_stats

        else:
            hard_agent = MCTSAgentHardPlayout("MCTSAgentHardPlayout", 0)
            mcts_agent = MCTSAgent("MinimaxAgent",  1)
            game = Game.create_agent_game(hard_agent, mcts_agent)
            game.init()
            game.start()

            winner = game.game_state.get_leader()

            if winner == 0:
                game_stats = [i, "HardPlayoutAgent", game.turns]
            else:
                game_stats = [i, "SoftPlayoutAgent", game.turns]

            games_df.loc[i] = game_stats

    games_df.to_csv("../data/MctsVsMctsHardPlayoutIterations.csv", index=False)

if __name__ == '__main__':
    run()