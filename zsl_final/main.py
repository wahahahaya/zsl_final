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
from build_loss import cpt_loss, ad_loss
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

    seed = 214
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

    optimizer = make_optimizer(config, model)
    scheduler = make_lr_scheduler(config, optimizer)

    use_mixed_precision = config.dtype == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    loss_cls = nn.CrossEntropyLoss().to(device)
    loss_ad = ad_loss(config.dataset_name).to(device)
    loss_cpt = cpt_loss(device).to(device)
    loss_reg = nn.MSELoss().to(device)

    losses = []
    cls_losses = []
    reg_losses = []
    ad_losses = []
    cpt_losses = []
    best_H = 0
    for epoch in range(0, config.epochs):
        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []
        ad_loss_epoch = []
        cpt_loss_epoch = []

        scheduler.step()
        for i, (img, att, label) in enumerate(train_dataloader):
            batch_img = img.to(device)
            batch_att = att.to(device)
            batch_label = label.to(device)

            score, part_feat, atten_map, atten_attr, query = model(batch_img, seen_att=att_seen, mode="train")

            Lcls = loss_cls(score, batch_label)
            Lreg = loss_reg(atten_attr, batch_att)
            Lad = loss_ad(query)
            Lcpt = loss_cpt(atten_map)

            loss = Lcls + Lreg + Lad + Lcpt

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_losses:
                scaled_losses.backward()
            optimizer.step()

            loss_epoch.append(loss.item())
            cls_loss_epoch.append(Lcls.item())
            reg_loss_epoch.append(Lreg.item())
            ad_loss_epoch.append(Lad.item())
            cpt_loss_epoch.append(Lcpt.item())

        losses += loss_epoch
        cls_losses += cls_loss_epoch
        reg_losses += reg_loss_epoch
        ad_losses += ad_loss_epoch
        cpt_losses += cpt_loss_epoch

        losses_mean = sum(losses) / len(losses)
        cls_losses_mean = sum(cls_losses) / len(cls_losses)
        reg_losses_mean = sum(reg_losses) / len(reg_losses)
        ad_losses_mean = sum(ad_losses) / len(ad_losses)
        cpt_losses_mean = sum(cpt_losses) / len(cpt_losses)

        print("Epoch: %d/%d, cls loss:%.4f, reg loss:%.4f, ad loss:%.4f, cpt loss:%.4f, loss:%.4f" % (
                epoch+1, config.epochs, cls_losses_mean, reg_losses_mean, ad_losses_mean, cpt_losses_mean, losses_mean
            )
        )

        synchronize()
        acc_train, acc_seen, acc_unseen, H = eval_gzsl(
            train_dataloader,
            test_seen_dataloader,
            test_unseen_dataloader,
            res,
            model,
            config.test_gamma,
            device
        )
        synchronize()

        print('train: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_train, acc_seen, acc_unseen, H))

        if H > best_H:
            best_epoch = epoch+1
            best_acc_train = acc_train
            best_acc_seen = acc_seen
            best_acc_unseen = acc_unseen
            best_H = H

    print("best: ep: %d" % best_epoch)
    print('train: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (best_acc_train, best_acc_seen, best_acc_unseen, best_H))


if __name__ == "__main__":
    config = parameter.get_parameters()
    log = config.__dict__
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.update({'time': time})
    if not os.path.isdir(config.tensorboard_dir):
        os.mkdir(config.tensorboard_dir)
    write_json(log, os.path.join(config.tensorboard_dir+"/config.json"))
    main(config)
