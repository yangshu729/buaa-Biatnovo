"""A wrapper class for scheduled optimizer """

import logging


logger = logging.getLogger(__name__)
class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model**-0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""
        self.n_steps += 1
        lr_scale = self._get_lr_scale()
        lr = self.lr_mul * lr_scale
        logger.info(f"step: {self.n_steps}, lr_scale:{lr_scale}, lr:{lr}")
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        """Return the state of the optimizer and scheduler"""
        return {
            'optimizer': self._optimizer.state_dict(),
            'lr_mul': self.lr_mul,
            'n_warmup_steps': self.n_warmup_steps,
            'n_steps': self.n_steps,
            'd_model': self.d_model,
        }

    def load_state_dict(self, state_dict):
        """Load the state of the optimizer and scheduler"""
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_mul = state_dict['lr_mul']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_steps = state_dict['n_steps']
        self.d_model = state_dict['d_model']
