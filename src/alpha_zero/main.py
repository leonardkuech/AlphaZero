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
    nnet = torch.load('../models/sugar_gliders_nnet1738687856.480237.pth')

    trainer = Trainer(nnet)
    trainer.learn(40,30)




if __name__ == '__main__':
    main()