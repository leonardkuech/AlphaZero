import numpy as np
import logging
import coloredlogs

from Trainer import Trainer
from CNN import GliderCNN
import cnn_tests

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')



def main():
    log.info('Starting Alpha Zero')

    # cnn_tests.cnn_loss_tests()
    nnet = GliderCNN()

    trainer = Trainer(nnet)
    trainer.learn(1,2)



if __name__ == '__main__':
    main()