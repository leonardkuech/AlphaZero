def calc_cantor(x,y):
    x = x * 2 if x >= 0 else -x * 2 - 1
    y = y * 2 if y >= 0 else -y * 2 - 1

    return ((x + y) * (x + y + 1)) // 2 + y
