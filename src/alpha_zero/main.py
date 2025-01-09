import numpy as np
import logging
import coloredlogs
import torch

from Trainer import Trainer
from CNN import GliderCNN
import cnn_tests

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

coloredlogs.install(level='INFO')


def main():
    log.info('Starting Alpha Zero')

    # cnn_tests.cnn_loss_tests()
    nnet = GliderCNN()

    trainer = Trainer(nnet)
    trainer.learn(100,3)




if __name__ == '__main__':
    main()