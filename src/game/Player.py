from Agent import Agent
from Cantor import calc_cantor


class Player:
    def __init__(self, player_id, name=None, agent: Agent = None):
        self.pos_x = 0
        self.pos_y = 0
        self.reserve = [0] * 5
        self.id = player_id
        self.name = name if name else (agent.name if agent else None)
        self.is_agent = agent is not None
        self.agent = agent

    def get_reserve(self):
        return self.reserve

    def reserve_is_empty(self):
        return all(amount <= 0 for amount in self.reserve)

    def set_reserve(self, reserve):
        self.reserve = reserve

    def get_id(self):
        return self.id

    def set_id(self, player_id):
        self.id = player_id

    def get_pos_y(self):
        return self.pos_y

    def set_pos_y(self, pos_y):
        self.pos_y = pos_y

    def get_pos_x(self):
        return self.pos_x

    def set_pos_x(self, pos_x):
        self.pos_x = pos_x

    def set_pos(self, pos_x, pos_y):
        self.pos_x = pos_x
        self.pos_y = pos_y

    def get_pos(self):
        return calc_cantor(self.pos_x, self.pos_y)

    def add_to_reserve(self, value):
        self.reserve[value - 1] += 1

    def subtract_from_reserve(self, value):
        self.reserve[value - 1] -= 1
        if self.reserve[value - 1] < 0:
            self.reserve[value - 1] = 0
            return False
        return True

    def subtract_lowest_from_reserve(self):
        for i, amount in enumerate(self.reserve):
            if amount > 0:
                self.reserve[i] -= 1
                return True
        return False

    def get_lowest_from_reserve(self):
        for i, amount in enumerate(self.reserve):
            if amount > 0:
                return i + 1
        return float('inf')

    def get_bank(self):
        return sum(amount * (i + 1) for i, amount in enumerate(self.reserve))

    def get_current_hex(self, game_board):
        return game_board.get_tile(c = calc_cantor(self.pos_x, self.pos_y))

    def set_agent(self, agent_flag):
        self.is_agent = agent_flag

    def choose_move(self, game_state):
        if self.agent:
            return self.agent.choose_move(game_state)
        return None

    def clone(self):
        cloned_player = Player(self.id, self.name)
        cloned_player.pos_x = self.pos_x
        cloned_player.pos_y = self.pos_y
        cloned_player.reserve = self.reserve[:]
        cloned_player.is_agent = self.is_agent
        return cloned_player

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
