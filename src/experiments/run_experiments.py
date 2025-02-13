import logging
import coloredlogs

import GreedyVSMCTS
import GreedyVSMinimax
import MctsVsMctsHardPlayout
import MctsVsMinimaxPruned
import MinimaxPrunedVsMinimaxPrunedAZEval
import MinimaxVsMcts
import MinimaxVsMinimaxPruned
import RandomVSMCTS
import RandomVSMinimax

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

coloredlogs.install(level='INFO')

def run():
    logger.info("Running experiments...")
    #RandomVSMinimax.run()
    logger.info("RandomVsMinimax ----> Done")
    #RandomVSMCTS.run()
    logger.info("RandomVsMcts ----> Done")
    #GreedyVSMinimax.run()
    logger.info("GreedyVsMinimax ----> Done")
    #GreedyVSMCTS.run()
    logger.info("GreedyVsMcts ----> Done")
    MinimaxVsMinimaxPruned.run()
    logger.info("MinimaxVsMinimaxPruned ----> Done")
    #MctsVsMctsHardPlayout.run()
    logger.info("MctsVsMctsHardPlay ----> Done")
    #MinimaxVsMcts.run()
    logger.info("MinimaxVsMcts ----> Done")
    MctsVsMinimaxPruned.run()
    logger.info("MctsVsMinimaxPruned ----> Done")
    MinimaxPrunedVsMinimaxPrunedAZEval.run()
    logger.info("MctsVsMinimaxPruned ----> Done")



if __name__ == '__main__':
    run()