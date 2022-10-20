# -*- coding: utf-8 -*-
""" main.py """

from configs.config import CFG
from model.ExampleModel import ExampleModel


def run():
    """Builds model, loads data, trains and evaluates"""
    model = ExampleModel(CFG)
    model.load_data()
    model.build()
    model.train()
    model.evaluate()


if __name__ == '__main__':
    run()
