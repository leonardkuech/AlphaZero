import pandas as pd
from numpy.ma.extras import average

from Game import Game
from RandomAgent import RandomAgent

def sim_branching():
    branch_df = pd.DataFrame()

    move_amounts = []

    for i in range(1000):

        move_amounts.append([])

        r_agent_1 = RandomAgent(name="r1")
        r_agent_2 = RandomAgent(name="r2")
        game = Game.create_agent_game(r_agent_1, r_agent_2)

        game.init()
        game.start()

        moves_agent1 = r_agent_1.amount_moves
        moves_agent2 = r_agent_2.amount_moves

        index = 0

        while index < len(moves_agent1):
            move_amounts[i].append(moves_agent1[index])

            if index < len(moves_agent2):
                move_amounts[i].append(moves_agent2[index])

            index += 1

    max_length = max(len(row) for row in move_amounts)

    # Pad shorter lists with None
    padded_data = [row + [None] * (max_length - len(row)) for row in move_amounts]

    df = pd.DataFrame(padded_data)

    df.to_csv('../data/branching.csv', index=False)

def calc_avg():

    branch_df = pd.read_csv('src/data/branching.csv')

    averages = branch_df.mean()

    averages.to_csv('../data/average_branching.csv', index=False)

def cut_first_entries():

    branch_df = pd.read_csv('src/data/average_branching.csv')

    entries_df = branch_df.head(100)

    entries_df.to_csv('../data/cut_first_entries.csv', index=False, header=False)

def main():
    sim_branching()
    calc_avg()
    cut_first_entries()







if __name__ == '__main__':
    main()