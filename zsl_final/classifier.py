import torch
import numpy as np
from sklearn.metrics import accuracy_score


def cal_accuracy(model, dataloader, att, test_id, device, bias=None):
    scores = []
    labels = []
    cpu = torch.device('cpu')

    for iteration, (img, attri, label) in enumerate(dataloader):
        img = img.to(device)
        score = model(img, seen_att=att, mode="test")
        scores.append(score)
        labels.append(label)

    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    if bias is not None:
        scores = scores-bias

    _,pred = scores.max(dim=1)
    pred = pred.view(-1).to(cpu)

    outpred = test_id[pred]
    outpred = np.array(outpred, dtype='int')
    labels = labels.numpy()
    unique_labels = np.unique(labels)
    acc = 0
    for ll in unique_labels:
        idx = np.nonzero(labels == ll)[0]
        acc += accuracy_score(labels[idx], outpred[idx])
    acc = acc / unique_labels.shape[0]
    return acc


def eval(
    train_loader,
    seen_loader,
    unseen_loader,
    att_seen,
    att_unseen,
    cls_seen_num,
    cls_unseen_num,
    train_id,
    gzsl_id,
    model,
    test_gamma,
    device
):
    acc_train = cal_accuracy(model=model, dataloader=train_loader, att=att_seen, test_id=train_id, device=device, bias=None)

    bias_s = torch.zeros((1, cls_seen_num)).fill_(test_gamma).to(device)
    bias_u = torch.zeros((1, cls_unseen_num)).to(device)
    bias = torch.cat([bias_s, bias_u], dim=1)

    att = torch.cat((att_seen, att_unseen), dim=0)
    acc_seen = cal_accuracy(model=model, dataloader=seen_loader, att=att, test_id=gzsl_id, device=device, bias=bias)
    acc_unseen = cal_accuracy(model=model, dataloader=unseen_loader, att=att, test_id=gzsl_id, device=device, bias=bias)

    H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)

    return acc_train, acc_seen, acc_unseen, H


def eval_gzsl(
    train_loader,
    seen_loader,
    unseen_loader,
    res,
    model,
    test_gamma,
    device
):
    model.eval()
    att_unseen = res['att_unseen'].to(device)
    att_seen = res['att_seen'].to(device)

    train_id = res['train_id']
    gzsl_id = res['train_test_id']

    cls_seen_num = att_seen.shape[0]
    cls_unseen_num = att_unseen.shape[0]

    with torch.no_grad():
        acc_train, acc_seen, acc_unseen, H = eval(
            train_loader,
            seen_loader,
            unseen_loader,
            att_seen,
            att_unseen,
            cls_seen_num,
            cls_unseen_num,
            train_id,
            gzsl_id,
            model,
            test_gamma,
            device
        )

    model.train()

    return acc_train, acc_seen, acc_unseen, H
