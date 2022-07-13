import json
from pathlib import Path
import torch
from torch.autograd import Variable
import numpy as np
import random


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def make_optimizer(config, model):

    lr = config.base_lr
    weight_decay = config.weight_decay
    momentum = config.momentum

    params_to_update = []
    params_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            params_names.append(name)

    optimizer = torch.optim.SGD(params_to_update, lr=lr,
                                weight_decay=weight_decay, momentum=momentum)

    return optimizer


def make_lr_scheduler(config, optimizer):
    step_size = config.steps
    gamma = config.gamma
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def generate_syn_feature(net_G, classes, attribute, num, device):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, 2048).to(device)
    syn_label = torch.LongTensor(nclass*num).to(device)
    syn_att = torch.FloatTensor(num, 312).to(device)
    syn_noise = torch.FloatTensor(num, 1024).to(device)

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = net_G(syn_noise, syn_att)
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


def compute_gradient_penalty(D, real_data, fake_data, attribute, device, lambda1):
    alpha = torch.rand(real_data.size(0), 1).to(device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True).to(device)

    disc_interpolates = D(interpolates, Variable(attribute))
    ones = torch.ones(disc_interpolates.size()).to(device)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda1
    return gradient_penalty


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
