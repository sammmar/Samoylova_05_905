# -*- coding: utf-8 -*-
"""Model config in json format"""
"""в конфигах мы определяем все, что можно настроить и изменить в будущем. 
Хорошими примерами являются гиперпараметры обучения, пути к папкам, архитектура модели, метрики, флаги.
Ваш эксперимент должен быть полностью описан здесь."""


CFG = {
    "data": {
        "path": "KMNIST",
        "image_size": 28,
        "load_with_info": True
    },
    "train": {
        "batch_size": 64,
        "nrof_epoch": 20,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy", "balanced_accuracy"]
    },
    "model": {
        "input": [28, 28, 1],
        "up_stack": {
            "layer_1": 128,
            "layer_2": 128,
        },
        "acivation_function": "relu",
        "output": 10
    }
}
