import math
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
"""
ref: https://github.com/mightydeveloper/Deep-Compression-PyTorch.git
"""
class PruningModule(Module):
    def prune_by_std(self, s=0.25):
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                module.prune(threshold)


class MaskedLinear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)




