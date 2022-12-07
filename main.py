# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from model.ExampleModel import ExampleModel


def run():
    """Builds model, loads data, trains and evaluates"""
    cfg = Config()

    trainer = Trainer(cfg)

    trainer.overfit_on_batch()


if __name__ == '__main__':
    run()
