# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py

class SchedulerOptim:
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps, max_lr, last_step):
        self.optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = last_step
        self.max_lr = max_lr

    def step(self):
        self.optimizer.step()
        self.update_learning_rate()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        return (self.d_model ** (-0.5)) * min(self.n_steps ** (-0.5), self.n_steps * (self.n_warmup_steps ** (-1.5)))

    def update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr_mul * self.get_lr_scale()
        if lr > self.max_lr:
            lr = self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
