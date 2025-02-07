import pandas as pd
from RandomAgent import RandomAgent
from Game import Game

def simulate_games():
    game_lengths = []

    for i in range(1000):

        r_agent_1 = RandomAgent(name="r1")
        r_agent_2 = RandomAgent(name="r2")
        game = Game.create_agent_game(r_agent_1, r_agent_2)

        game.init()
        game.start()

        game_lengths.append(game.turns)

    df = pd.DataFrame(game_lengths)
    average_length_df = df.mean()

    df.to_csv('../data/game_length.csv', index=False)
    average_length_df.to_csv('../data/game_length_average.csv', index=False)


def main():
    simulate_games()

if __name__ == '__main__':
    main()