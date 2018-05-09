

import torch
import math



def split_data(input, target, ratio=0.8):
    split_idx = int(input.size(0) * ratio)
    end_idx = input.size(0) + 1
    return input[0:split_idx,], target[0:split_idx,], input[split_idx:end_idx, ], target[split_idx:end_idx, ]


def data_generator(ratio=0.8, normalized=False):
    data = torch.FloatTensor(1000, 2).uniform_(0, 1) - 0.5
    distance = torch.sqrt(torch.pow(data[:, 0], 2) + torch.pow(data[:, 1], 2)).view(-1, 1)
    radius = 1 / math.sqrt(2 * math.pi)
    inside = distance.clone().apply_(lambda x: 1 if x < radius else -1)
    outside = distance.clone().apply_(lambda x: 1 if x > radius else -1)
    target = torch.cat((inside, outside), 1)
    if normalized:
        data = (data - data.mean()) / data.std()

    return split_data(data, target, ratio)