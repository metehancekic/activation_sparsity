import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn


# def DReLU(x, bias=0, filters=None, epsilon=8.0/255):

#     def bias_calculator(filters, epsilon):
#         # breakpoint()
#         bias = epsilon * torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
#         bias = bias.unsqueeze(dim=2)
#         bias = bias.unsqueeze(dim=3)
#         return bias

#     if isinstance(filters, torch.Tensor):
#         bias = bias_calculator(filters, epsilon)

#     # breakpoint()
#     return F.relu(x - bias) - F.relu(-x - bias)


# def DTReLU(x, bias=0, filters=None, epsilon=8.0/255):

#     def bias_calculator(filters, epsilon):
#         # breakpoint()
#         bias = epsilon * torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
#         bias = bias.unsqueeze(dim=2)
#         bias = bias.unsqueeze(dim=3)
#         return bias

#     if isinstance(filters, torch.Tensor):
#         bias = bias_calculator(filters, epsilon)
#     return F.relu(x - bias) + bias * torch.sign(F.relu(x - bias)) - F.relu(-x - bias) - bias * torch.sign(F.relu(-x - bias))


class DReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def bias_calculator(self, filters):
        bias = torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    def forward(self, x, filters, alpha=8./255):
        bias = alpha * self.bias_calculator(filters)
        return F.relu(x - bias) - F.relu(-x - bias)

    def __repr__(self):
        s = f"DReLU()"
        return s.format(**self.__dict__)


class DTReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def bias_calculator(self, filters):
        bias = torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    def forward(self, x, filters, alpha=8./255):
        bias = alpha * self.bias_calculator(filters)
        return F.relu(x - bias) + bias * torch.sign(F.relu(x - bias)) - F.relu(-x - bias) - bias * torch.sign(F.relu(-x - bias))
        #F.threshold(x, threshold=bias, value=0.) - F.threshold(-x, threshold=bias, value=0.)

    def __repr__(self):
        s = f"DReLU()"
        return s.format(**self.__dict__)


class TReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def bias_calculator(self, filters):
        bias = torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    def forward(self, x, filters, alpha=8./255):
        bias = alpha * self.bias_calculator(filters)
        return F.relu(x - bias) + bias * torch.sign(F.relu(x - bias))

    def __repr__(self):
        s = f"TReLU()"
        return s.format(**self.__dict__)


class TReLU_with_trainable_bias(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.bias = Parameter(torch.randn((1, in_channels, 1, 1))/10., requires_grad=True)
        torch.nn.init.xavier_normal_(self.bias)
        self.register_parameter("bias", self.bias)

    def forward(self, x):
        return F.relu(x - torch.abs(self.bias)) + torch.abs(self.bias) * torch.sign(F.relu(x - torch.abs(self.bias)))

    def __repr__(self):
        s = f"TReLU_with_trainable_bias(in_channels = {self.in_channels})"
        return s.format(**self.__dict__)


def test_DReLU():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(-10, 10, 0.1), DReLU(torch.Tensor(np.arange(-10, 10, 0.1)), bias=5))
    plt.savefig("double_sided_relu")


if __name__ == '__main__':
    test_DReLU()
