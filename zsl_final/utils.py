import json
from pathlib import Path
import torch


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
