import logging
import math
import torch

from CNN import GliderCNN as cnn
from GameState import GameState
from Utils import MOVE_TO_INDEX

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

EPS = 1e-8

def evaluate(game_state: GameState):
    winner = game_state.get_leader()
    if winner < 0:
        return -1.0  # Tie
    return 1.0 if winner == game_state.player_to_move else -1.0


class MCTS():

    def __init__(self, nnet: cnn, exp_const = 0.1, simulations = 150):
        self.nnet = nnet
        self.Qsa = {}  # Expected reward taking an action a from a GameState s
        self.Nsa = {}  # Number of times action a was taken from GameState s
        self.Ns = {} # Number of times board s was visited
        self.Ps = {}  # Stores initial policy from neural net

        self.SIMULATION_LIMIT = simulations  # Number of simulations per move
        self.EXPLORATION_CONSTANT = exp_const  # UCT exploration constant

    def get_action_probabilities(self, game_state: GameState):

        for i in range(self.SIMULATION_LIMIT):
            self.search(game_state)

        probabilities = torch.zeros(len(MOVE_TO_INDEX))

        s = game_state.string_representation()
        sum = 0
        for move in game_state.get_moves():
            sum += self.Nsa.get((s, move),0)
            probabilities[MOVE_TO_INDEX[move]] = self.Nsa.get((s, move),0) / (self.Ns[s])

        return probabilities


    def search(self, game_state: GameState):

        log.debug(f'Current player is {game_state.player_to_move}')
        if game_state.check_game_over():
            log.debug(f'Game over and winner is {game_state.get_leader()}')
            log.debug(f'Returning reward {- evaluate(game_state)}')
            return  - evaluate(game_state)

        s = game_state.string_representation()

        if s not in self.Ns: # Leaf Node
            log.debug(f'Leaf Node reached')
            self.Ns[s] = 0
            board, player1, player2 = game_state.encode()
            self.Ps[s], v = self.nnet.predict(board, player1, player2)
            moves = game_state.get_moves()
            move_tensor = torch.zeros(len(MOVE_TO_INDEX))
            for move in moves:
                move_tensor[MOVE_TO_INDEX[move]] = 1
            self.Ps[s] *= move_tensor
            self.Ps[s] /= self.Ps[s].sum()

            return - v

        log.debug(f'Finding move with highest uct')
        log.debug(f' {s} ---> {self.Ps[s]}')
        max_u, best_a = -float("inf"), -float("inf")
        for a in game_state.get_moves():
            qsa = self.Qsa.get((s,a), 0)
            nsa = self.Nsa.get((s,a), 0)
            u = qsa + self.EXPLORATION_CONSTANT * self.Ps[s][0, MOVE_TO_INDEX[a]] * math.sqrt(self.Ns[s] + EPS) / (
                        1 + nsa)
            log.debug(f'Move {a} (index = {MOVE_TO_INDEX[a]}) has uct = {u} and was visited {self.Nsa.get((s, a), 0)} times')
            #log.debug(f'Values qsa = {qsa}, nsa = {nsa}, self.Ps[s][0, MOVE_TO_INDEX[a]] = {self.Ps[s][0, MOVE_TO_INDEX[a]]}, math.sqrt(self.Ns[s] + EPS) = {math.sqrt(self.Ns[s] + EPS)}')
            if u > max_u:
                max_u = u
                best_a = a

        a = best_a
        log.debug(f'Best move: {a}, which was visited {self.Nsa.get((s, a), 0)} times')

        next_state = game_state.clone()
        next_state.apply_move(a)

        v = self.search(next_state)

        self.Ns[s] += 1
        self.Nsa[(s,a)] = self.Nsa.get((s,a), 0) + 1
        self.Qsa[(s,a)] = (self.Nsa[(s,a)] * self.Qsa.get((s,a), 0) + v) / (self.Nsa[(s,a)] + 1)

        return  - v

    def mask(self, moves):
        pass