import torch

from NNetAgent import NNetAgent
from Game import Game
import logging
from CNN import *

logger = logging.getLogger(__name__)

def main():
    games_won = 0
    player_id_new_agent = 0

    for i in range(1000):
        old_nnet = torch.load("../models/sugar_gliders_nnet1738568988.3762062.pth")
        new_nnet = torch.load("../models/sugar_gliders_nnet1738687856.480237.pth")
        agent_old = NNetAgent(old_nnet, "OldAgent")
        agent_new = NNetAgent(new_nnet, "NewAgent")

        if i % 2 == 0:
            game = Game.create_agent_game(agent_old, agent_new)
            player_id_new_agent = 1
        else:
            game = Game.create_agent_game(agent_new, agent_old)
            player_id_new_agent = 0

        game.init()
        game.start()
        leader = game.game_state.get_leader()

        if leader < 0:
            print('Draw')
        else:
            print(f'Winner: {leader} Name: {game.game_state.players[leader].get_name()}')

        if leader == player_id_new_agent:
            print('+')
            games_won += 1

    logger.info(f'Played {100} games and won {games_won}')
    print(f'Played {1000} games and won {games_won}')


if __name__ == '__main__':
    main()