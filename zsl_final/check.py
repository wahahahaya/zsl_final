from datetime import datetime
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from os.path import join
import pickle
from apex import amp

import parameter
import build_data
import build_model
from build_loss import cpt_loss
from resnet_feature import resnet101_features
from classifier import eval_gzsl
from utils import write_json, make_optimizer, make_lr_scheduler


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # seed = random.randint(1, 10000)
    seed = 7384
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataloader, test_seen_dataloader, test_unseen_dataloader, res = build_data.build_dataloader(config)
    att_seen = res['att_seen'].to(device)
    res101 = resnet101_features(pretrained=True)

    w2v_file = config.dataset_name+"_attribute.pkl"
    w2v_path = join("/HDD-1_data/arlen/zsl_final/datasets/Attribute/w2v/", w2v_file)
    with open(w2v_path, 'rb') as f:
        w2v = pickle.load(f)
    w2v = torch.from_numpy(w2v).float().to(device)
    model = build_model.DSACA_Net(res101, w2v).to(device)
    model.load_state_dict(torch.load("/HDD-1_data/arlen/zsl_final/log/model/SUN_348_24_7384.pth"))

    acc_train, acc_seen, acc_unseen, H = eval_gzsl(
        train_dataloader,
        test_seen_dataloader,
        test_unseen_dataloader,
        res,
        model,
        config.test_gamma,
        device
    )

    print('train: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_train, acc_seen, acc_unseen, H))


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    config = parameter.get_parameters()
    # log = config.__dict__
    # time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # log.update({'time': time})
    # if not os.path.isdir(config.tensorboard_dir):
    #     os.mkdir(config.tensorboard_dir)
    # write_json(log, os.path.join(config.tensorboard_dir+"/config_7384_AWA2.json"))
    # for _ in range(30):
    main(config)
