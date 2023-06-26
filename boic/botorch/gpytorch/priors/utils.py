import math
import torch

def normal_prob(value):
    return 1 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * value.pow(2))


def normal_log_prob(value):
    return -0.5 * (math.log(2 * math.pi) + value.pow(2))


def normal_cdf(value):
    return 0.5 * (1 + torch.erf(value / math.sqrt(2)))


def inverse_normal_cdf(value):
    return torch.erfinv(2 * value - 1) * math.sqrt(2)