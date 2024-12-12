from Cantor import calc_cantor


class HexTile:
    def __init__(self, x, y, z, value=0):
        self.x = x
        self.y = y
        self.z = z
        self.value = value

        self.valid = self._check_valid_hex_tile()
        self.is_occupied = False

    def _check_valid_hex_tile(self):
        return self.x + self.y + self.z == 0

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_z(self, z):
        self.z = z

    def is_valid(self):
        return self.valid

    def check_fruit_exists(self):
        return self.value > 0

    def __eq__(self, other):
        if not isinstance(other, HexTile):
            return False
        return self.x == other.get_x() and self.y == other.get_y() and self.z == other.get_z()

    @staticmethod
    def subtract(a, b):
        return [a.x - b.x, a.y - b.y, a.z - b.z]

    @staticmethod
    def add(a, b):
        return [a.get_x() + b.get_x(), a.get_y() + b.get_y(), a.get_z() + b.get_z()]

    @staticmethod
    def get_distance(a, b):
        distance = HexTile.subtract(a, b)
        for value in distance:
            if value != 0:
                return abs(value)
        return 0

    def get_cantor(self):
        return calc_cantor(self.x, self.y)

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def set_occupied(self, occupied):
        self.is_occupied = occupied

    def clone(self):
        cloned_tile = HexTile(self.x, self.y, self.z, self.value)
        cloned_tile.set_occupied(self.is_occupied)
        return cloned_tile

