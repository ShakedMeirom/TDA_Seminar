###### Slayer Implementations ########

# This module allows to choose the specific implementation of the PH input layer.

import torch
from torch import nn


def adjust_dimensions(param, max_points, point_dimension, batch_size):
    adj_param = torch.cat([param] * max_points, 1)
    adj_param = adj_param.view(-1, point_dimension)
    adj_param = torch.stack([adj_param] * batch_size, 0)
    return adj_param


def Slayer_Exponential(Slayer, batch, batch_size, max_points, not_dummy_points):
    # centers = torch.cat([Slayer.centers] * max_points, 1)
    # centers = centers.view(-1, Slayer.point_dimension)
    # centers = torch.stack([centers] * batch_size, 0)
    #
    # sharpness = torch.cat([Slayer.sharpness] * max_points, 1)
    # sharpness = sharpness.view(-1, Slayer.point_dimension)
    # sharpness = torch.stack([sharpness] * batch_size, 0)

    centers = adjust_dimensions(Slayer.centers, max_points, Slayer.point_dimension, batch_size)
    sharpness = adjust_dimensions(Slayer.sharpness, max_points, Slayer.point_dimension, batch_size)

    x = centers - batch
    x = x.pow(2)
    x = torch.mul(x, sharpness)
    x = torch.sum(x, 2)
    x = torch.exp(-x)
    x = torch.mul(x, not_dummy_points)
    x = x.view(batch_size, Slayer.n_elements, -1)
    x = torch.sum(x, 2)
    x = x.squeeze()

    return x


def Slayer_Avg_Exponential(Slayer, batch, batch_size, max_points, not_dummy_points):
    centers = adjust_dimensions(Slayer.centers, max_points, Slayer.point_dimension, batch_size)

    sharpness = adjust_dimensions(Slayer.sharpness, max_points, Slayer.point_dimension, batch_size)

    mult_sharpness = adjust_dimensions(Slayer.mult_sharpness, max_points, 1, batch_size)
    mult_sharpness = mult_sharpness.squeeze()

    avg_weights_init = adjust_dimensions(Slayer.avg_weights_init, max_points, 1, batch_size)
    avg_weights_init = avg_weights_init.squeeze()

    x = centers - batch

    x = x.pow(2)
    x = torch.mul(x, sharpness)
    x = torch.sum(x, 2)

    if Slayer.point_dimension > 1:
        x = x + torch.mul(torch.mul(batch[:,:,0], batch[:,:,1]), mult_sharpness)

    x = torch.exp(-x)
    x = torch.mul(x, avg_weights_init)
    x = torch.mul(x, not_dummy_points)
    x = x.view(batch_size, Slayer.n_elements, -1)
    x = torch.sum(x, 2)
    x = x.squeeze()

    return x


