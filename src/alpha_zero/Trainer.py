import numpy as np
import logging

import torch

from CNN import GliderCNN
from Game import Game
from MCTS import MCTS
from NNetAgent import NNetAgent

logger = logging.getLogger(__name__)

class Trainer:
    PIT_GAMES = 100
    THRESHOLD = 0.55

    def __init__(self, nnet: GliderCNN):

        self.nnet = nnet
        self.training_examples= []

    def learn(self, iterations, games):
        logger.info('Start Learning...')

        for i in range(iterations):
            logger.info(f'Starting Iteration #{i} ...')

            for j in range(games):
                logger.info(f'Starting Game #{j} ...')
                self.play()

            new_nnet = self.trainNNet()
            percentage_won = self.pit(self.nnet, new_nnet)

            if percentage_won > Trainer.THRESHOLD:
                self.nnet = new_nnet

    # TODO save nnet


    def play(self):

        game = Game.create_game().game_state
        trainExamples = []
        self.curPlayer = 0
        episodeStep = 0

        mcts = MCTS(self.nnet)

        while True:
            prob = mcts.get_action_probabilities(game)
            trainExamples.append([game.encode(), prob, None])

            if game.check_game_over():
                winner = game.get_leader() # assign winner
                for i, sample in enumerate(trainExamples):
                    if winner < 0:
                        sample[2] = torch.tensor(0.5)
                    else:
                        sample[2] = torch.tensor(1) if i % 2 == winner else torch.tensor(0)
                break

        self.training_examples.extend(trainExamples)


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

            game.start()

            if game.game_state.get_leader() == player_id_new_agent:
                games_won += 1

        return games_won / Trainer.PIT_GAMES
