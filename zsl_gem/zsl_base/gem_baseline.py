import torch
import build
import random
import GEM_model
import GEM_train
import numpy as np


def make_optimizer(model):
    lr = 1e-3
    weight_decay = 1e-5
    momentum = 0.9

    params_to_update = []
    params_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            params_names.append(name)

    optimizer = torch.optim.SGD(params_to_update, lr=lr,
                                weight_decay=weight_decay, momentum=momentum)

    return optimizer


def make_lr_scheduler(optimizer):
    step_size = 10
    gamma = 0.5
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GEM_model.build_GEMNet().to(device)

    optimizer = make_optimizer(model)
    scheduler = make_lr_scheduler(optimizer)

    tr_dataloader, tu_loader, ts_loader, res = build.build_dataloader(is_distributed=False)

    lamd = {
        1: 1.0,
        2: 0.05,
        3: 0.2,
    }

    GEM_train.do_train(
        model,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        0.7,
        device,
        25,
    )

    return model


def main():
    torch.backends.cudnn.benchmark = True
    manualSeed = 2022
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    train_model()


if __name__ == '__main__':
    main()
