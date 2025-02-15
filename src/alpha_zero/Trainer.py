import numpy as np
import logging

import torch
from sympy import false

from CNN import GliderCNN
from Game import Game
from Utils import INDEX_TO_MOVE, sum_reserve
from MCTS import MCTS
from NNetAgent import NNetAgent
import time

logger = logging.getLogger(__name__)

class Trainer:
    PIT_GAMES = 100
    THRESHOLD = 0.55

    def __init__(self, nnet: GliderCNN):
        self.nnet = torch.compile(nnet, mode='max-autotune')
        self.training_examples= []

    def learn(self, iterations, games):
        logger.info('Start Learning...')

        for i in range(iterations):
            logger.info(f'Starting Iteration #{i} ...')

            for j in range(games):
                self.play()
                logger.info(f'Game #{j} done')

            logger.info(f'Finished self play')
            logger.info(f'Currently {len(self.training_examples)} Samples ')

            new_nnet = self.nnet.trainCNN(self.training_examples)
            new_nnet = new_nnet.trainCNN(self.training_examples)
            percentage_won = self.pit(self.nnet, new_nnet)

            if percentage_won >=Trainer.THRESHOLD:
                logger.info('Updated CNN to next Generation')
                self.nnet = new_nnet
                torch.save(self.nnet, f'../models/sugar_gliders_nnet{time.time()}.pth')
                self.training_examples = []


    def play(self):

        game = Game.create_game().game_state
        train_examples = []
        mcts = MCTS(self.nnet)

        turns = 0
        while True:
            prob = mcts.get_action_probabilities(game)
            #logger.info(f'{prob}')
            prob /= np.sum(prob)
            train_examples.append([game.encode(), prob.reshape(1,62), None])

            valid = false
            move = None

            while not valid:
                index = np.random.choice(len(prob), p=prob)
                move = INDEX_TO_MOVE[index]

                valid = game.apply_move(move)
                turns += 1

            #logger.info(f'{move}')
            if game.check_game_over():
                winner = game.get_leader() # assign winner
                for i, sample in enumerate(train_examples):
                    if winner < 0:
                        sample[2] = torch.tensor([-1])
                    else:
                        sample[2] = torch.tensor([1]) if i % 2 == winner else torch.tensor([-1])
                break

        self.training_examples.extend(train_examples)


    def pit(self, old_nnet, new_nnet) -> float:

        games_won = 0
        player_id_new_agent = 0

        for i in range(Trainer.PIT_GAMES):
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
                games_won += 0.5
                logger.info(f'Draw | {sum_reserve(game.game_state.reserves[0])} Points (Player 1) against {sum_reserve(game.game_state.reserves[1])} points (Player 2) in {game.turns}')
            else:
                logger.info(f'Player {leader} won game #{i} | {sum_reserve(game.game_state.reserves[0])} Points (Player 1) against {sum_reserve(game.game_state.reserves[1])} points (Player 2) {game.turns}')

            if leader == player_id_new_agent:
                games_won += 1

        logger.info(f'Player {self.PIT_GAMES} games and won {games_won}')

        return games_won / Trainer.PIT_GAMES
