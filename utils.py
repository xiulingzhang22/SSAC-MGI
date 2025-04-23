import os
import shutil

import numpy as np
import torch as t

import math

def log_parameters(log_dir_path, parameters):
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    else:
        shutil.rmtree(log_dir_path)
        os.makedirs(log_dir_path)

    f_path = log_dir_path + '/params.txt'
    f = open(f_path, 'w+')

    for k, v in parameters.items():
        f.write(f'{k}:{v.value}\n')
        f.flush()

def create_exp_logfile(logdir_path):
    if not os.path.exists(logdir_path):
        os.makedirs(logdir_path)
    
    f_path = logdir_path + '/results_test.csv'

    if os.path.isfile(f_path):
        os.remove(f_path)

    return open(f_path, 'w+')

def performance_avg(episodes, k):
    return np.mean(episodes[-k:])

#------------------------------------------------------------

# TRPO utils from https://github.com/ikostrikov/pytorch-trpo


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * t.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = t.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = t.cat(grads)
    return flat_grad

#--------------------------------------------------------------