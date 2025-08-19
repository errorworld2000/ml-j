import torch
from mini_ml.utils.register import LR_SCHEDULERS

@LR_SCHEDULERS.register()
class PolynomialDecay:
    """
    多项式衰减学习率
    lr = (learning_rate - end_lr) * (1 - cur_iter / max_iter) ** power + end_lr
    """
    def __init__(self, optimizer: torch.optim.Optimizer, learning_rate, end_lr=0.0, power=0.9, max_iters=10000):
        self.optimizer = optimizer
        self.lr_start = learning_rate
        self.lr_end = end_lr
        self.power = power
        self.max_iters = max_iters
        self.last_iter = 0

    def step(self):
        """更新当前迭代步数并修改学习率"""
        self.last_iter += 1
        lr = (self.lr_start - self.lr_end) * (1 - self.last_iter / self.max_iters) ** self.power + self.lr_end
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]