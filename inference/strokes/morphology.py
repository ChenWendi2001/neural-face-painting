import torch.nn.functional as F


def erosion(x, m=1):
    return -F.max_pool2d(
        -x, kernel_size=2 * m + 1, padding=m, stride=1)


def dilation(x, m=1):
    return F.max_pool2d(
        x, kernel_size=2 * m + 1, padding=m, stride=1)
