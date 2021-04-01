import torch

def weights_init_normal(module):
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(module.weight.data, 1.0, 0.02)
        torch.nn.init.constant(module.bias.data, 0.0)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert n_epochs > decay_start_epoch, 'Decay must start before training session ends'
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        den = self.n_epochs - self.decay_start_epoch
        num = epoch + self.offset - self.decay_start_epoch
        num = max(num, 0)
        return 1 - num/den