import GreedyVSMCTS
import GreedyVSMinimax
import MctsVsMctsHardPlayout
import MctsVsMinimaxPruned
import MinimaxVsMcts
import MinimaxVsMinimaxPruned
import RandomVSMCTS
import RandomVSMinimax


def run():
    RandomVSMinimax.run()
    RandomVSMCTS.run()
    GreedyVSMinimax.run()
    GreedyVSMCTS.run()
    MinimaxVsMinimaxPruned.run()
    MctsVsMctsHardPlayout.run()
    MinimaxVsMcts.run()
    MctsVsMinimaxPruned.run()



if __name__ == '__main__':
    run()