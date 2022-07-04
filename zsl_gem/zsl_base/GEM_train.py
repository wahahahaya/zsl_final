import torch
import torch.distributed as dist
from inferencer import eval_zs_gzsl
from torch.utils.tensorboard import SummaryWriter


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
        model,
        tr_dataloader,
        tu_loader,
        ts_loader,
        res,
        optimizer,
        scheduler,
        lamd,
        test_gamma,
        device,
        max_epoch,):

    tensorboard_sw = False
    if tensorboard_sw:
        train_dir = 'tensorboard_atten/class_gem_ver1'
        train_writer = SummaryWriter(log_dir=train_dir)

    best_performance = [0, 0, 0, 0]
    best_epoch = -1

    att_seen = res['att_seen'].to(device)

    losses = []
    cls_losses = []
    reg_losses = []
    ad_losses = []
    cpt_losses = []
    scale_all = []

    model.train()

    for epoch in range(0, max_epoch):

        loss_epoch = []
        cls_loss_epoch = []
        reg_loss_epoch = []
        ad_loss_epoch = []
        cpt_loss_epoch = []

        scale_epoch = []

        scheduler.step()

        for iteration, (batch_img, batch_att, batch_label) in enumerate(tr_dataloader):
            batch_img = batch_img.to(device)
            batch_att = batch_att.to(device)
            batch_label = batch_label.to(device)

            loss_dict = model(x=batch_img, att=batch_att, label=batch_label, seen_att=att_seen,)

            Lreg = loss_dict['Reg_loss']
            Lcls = loss_dict['Cls_loss']
            Lad = loss_dict['AD_loss']
            Lcpt = loss_dict['CPT_loss']

            scale = loss_dict['scale']

            loss_dict.pop('scale')

            # loss = Lcls + lamd[1]*Lreg + lamd[2]*Lad + lamd[3]*Lcpt
            loss = Lcls + lamd[1]*Lreg + lamd[2]*Lad

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.append(loss.item())
            cls_loss_epoch.append(Lcls.item())
            reg_loss_epoch.append(Lreg.item())
            ad_loss_epoch.append(Lad.item())
            cpt_loss_epoch.append(Lcpt.item())
            scale_epoch.append(scale)

        if is_main_process():
            losses += loss_epoch
            cls_losses += cls_loss_epoch
            reg_losses += reg_loss_epoch
            ad_losses += ad_loss_epoch
            cpt_losses += cpt_loss_epoch
            scale_all += scale_epoch

            loss_epoch_mean = sum(loss_epoch)/len(loss_epoch)
            cls_loss_epoch_mean = sum(cls_loss_epoch)/len(cls_loss_epoch)
            reg_loss_epoch_mean = sum(reg_loss_epoch)/len(reg_loss_epoch)
            ad_loss_epoch_mean = sum(ad_loss_epoch)/len(ad_loss_epoch)
            cpt_loss_epoch_mean = sum(cpt_loss_epoch)/len(cpt_loss_epoch)
            scale_epoch_mean = sum(scale_epoch) / len(scale_epoch)

            losses_mean = sum(losses) / len(losses)
            cls_losses_mean = sum(cls_losses) / len(cls_losses)
            reg_losses_mean = sum(reg_losses) / len(reg_losses)
            ad_losses_mean = sum(ad_losses) / len(ad_losses)
            cpt_losses_mean = sum(cpt_losses) / len(cpt_losses)
            scale_all_mean = sum(scale_all) / len(scale_all)

            log_info = 'epoch: %d  |  loss: %.4f (%.4f), cls_loss: %.4f (%.4f),   reg_loss: %.4f (%.4f),   ad_loss: %.4f (%.4f),   cpt_loss: %.4f (%.4f),   scale:  %.4f (%.4f),    lr: %.6f' % \
                       (epoch + 1, loss_epoch_mean, losses_mean, cls_loss_epoch_mean, cls_losses_mean, reg_loss_epoch_mean,
                        reg_losses_mean, ad_loss_epoch_mean, ad_losses_mean, cpt_loss_epoch_mean, cpt_losses_mean,
                        scale_epoch_mean, scale_all_mean, optimizer.param_groups[0]["lr"])
            print(log_info)

        synchronize()
        acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
            tu_loader,
            ts_loader,
            res,
            model,
            test_gamma,
            device)

        synchronize()

        if is_main_process():
            print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_zs, acc_seen, acc_novel, H))

            if H > best_performance[-1]:
                best_epoch = epoch+1
                best_performance[1:] = [acc_seen, acc_novel, H]

            if acc_zs > best_performance[0]:
                best_performance[0] = acc_zs

        if tensorboard_sw:
            train_writer.add_scalar("CLS loss", cls_losses_mean, epoch)
            train_writer.add_scalar("Reg loss", reg_losses_mean, epoch)
            train_writer.add_scalar("Ad loss", ad_losses_mean, epoch)
            train_writer.add_scalar("Cpt loss", cpt_losses_mean, epoch)
            train_writer.add_scalar("scale", scale_epoch_mean, epoch)
            train_writer.add_scalar("seen acc", acc_seen, epoch)
            train_writer.add_scalar("unseen acc", acc_novel, epoch)
            train_writer.add_scalar("H", H, epoch)

    if is_main_process():
        print("best: ep: %d" % best_epoch)
        print('zsl: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % tuple(best_performance))
