import GreedyVSMCTS
import GreedyVSMinimax
import RandomVSMCTS
import RandomVSMinimax


def run():
    RandomVSMinimax.run()
    RandomVSMCTS.run()
    GreedyVSMinimax.run()
    GreedyVSMCTS.run()


if __name__ == '__main__':
    run()