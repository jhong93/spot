"""
From "A Context-Aware Loss Function for Action Spotting in Soccer Videos"
"""

import numpy as np
import torch
import torch.nn.functional as F


class ContextAwareWeights:

    def __init__(self, k1=2, k2=1, k3=2, k4=2, hit_radius=0.1, miss_radius=0.9):
        n = k1 + k2 + k3 + k4
        mul_w = np.ones(n)
        add_w = np.ones(n)
        radius = np.full(n, miss_radius)
        for i in range(n):
            if i < k1:
                mul_w[i] = (k1 - i) / k1
            elif i < k1 + k2:
                mul_w[i] = 0
            elif i < k1 + k2 + k3:
                mul_w[i] = ((i - k1 - k2) - k3) / k3
                add_w[i] = (i - k1 - k2) / k3
                radius[i] = 1. - hit_radius
            else:
                mul_w[i] = (i - k1 - k2 - k3) / k4
        self._w = np.stack([mul_w, add_w, radius], axis=1)
        self._offset = k1 + k2

    @property
    def weights(self):
        return self._w

    @property
    def offset(self):
        return self._offset

    def __len__(self):
        return self._w.shape[0]


CALF_ERROR_FLAG = True


def set_calf_error_flag():
    global CALF_ERROR_FLAG
    CALF_ERROR_FLAG = 1


def get_calf(pred, weights):
    pred_scores = F.softmax(pred, dim=2)    # (N, L, C)
    cl = -torch.log(
        weights[:, :, :, 1] - pred_scores[:, :, 1:] * weights[:, :, :, 0]
    ) + torch.log(weights[:, :, :, 2])
    cl = torch.max(torch.zeros_like(cl), cl)

    global CALF_ERROR_FLAG
    if CALF_ERROR_FLAG:
        tmp = torch.sum(cl)
        if torch.isinf(tmp):
            print('Found Inf in CALF. Supressing future errors.')
            CALF_ERROR_FLAG = False
        if torch.isnan(tmp):
            print('Found NaN in CALF. Supressing future errors.')
            CALF_ERROR_FLAG = False
    return torch.mean(cl)


if __name__ == '__main__':
    c = ContextAwareWeights()
    print(c.weights)
    print('All 1:', -(c.weights[:, 1] - np.ones(len(c)) * c.weights[:, 0]))
    print('All 0:', -(c.weights[:, 1] - np.zeros(len(c)) * c.weights[:, 0]))