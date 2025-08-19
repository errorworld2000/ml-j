from torch import optim
from mini_ml.utils.register import OPTIMIZERS


@OPTIMIZERS.register()
class SGD:
    """SGD 优化器"""

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.optimizer = optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    def get(self):
        return self.optimizer


@OPTIMIZERS.register()
class Adam:
    """Adam 优化器"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0):
        self.optimizer = optim.Adam(
            params, lr=lr, betas=betas, weight_decay=weight_decay
        )

    def get(self):
        return self.optimizer


@OPTIMIZERS.register()
class AdamW:
    """AdamW 优化器"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01):
        self.optimizer = optim.AdamW(
            params, lr=lr, betas=betas, weight_decay=weight_decay
        )

    def get(self):
        return self.optimizer
