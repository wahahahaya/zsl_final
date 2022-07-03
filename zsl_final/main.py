import torch
import random
import model
import data
import classifier
import numpy as np
import inferencer
from numpy import genfromtxt
from resnet_feature import resnet101_features
from sklearn.metrics import accuracy_score


lamd = {
    1: 1.0,
    2: 0.05,
    3: 0.2,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

manualSeed = 2022
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

data_loader = data.DATA_LOADER("CUB")

train_loader = data_loader.train_loader
seen_loader = data_loader.seen_loader
unseen_loader = data_loader.unseen_loader

att_seen = data_loader.attribute_seen.to(device)
att_unseen = data_loader.attribute_unseen.to(device)
at_map = torch.cat((att_seen, att_unseen), dim=0).to(device)
train_id = data_loader.train_id
train_text_id = data_loader.train_test_id

data_set_path = '../dataset/'
glove_path = data_set_path + "glove_embedding.csv"
w2v = genfromtxt(glove_path, delimiter=',', skip_header=0)
w2v = torch.from_numpy(w2v).float().to(device)

res101 = resnet101_features(pretrained=True).to(device)

net = model.DSACAN(res101, w2v).to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

loss_reg = torch.nn.MSELoss()
loss_cls = torch.nn.CrossEntropyLoss()


def cal_acc(loader, net, see_attri, test_id, bias=None):
    scores = []
    labels = []
    for iteration, (feature, att, label) in enumerate(loader):
        feature = feature.to(device)

        pred_att, pred_class = net(feature, see_attri)
        scores.append(pred_class)
        labels.append(label)

    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    if bias is not None:
        scores = scores-bias

    _, pred = scores.max(dim=1)
    pred = pred.view(-1).cpu()

    outpred = test_id[pred]
    outpred = np.array(outpred, dtype='int')
    labels = labels.numpy()
    unique_labels = np.unique(labels)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(labels == l)[0]
        acc += accuracy_score(labels[idx], outpred[idx])
    acc = acc / unique_labels.shape[0]
    return acc


losses = []
best_gzsl_acc = 0
for epoch in range(0, 50):
    loss_epoch = []
    for iteration, (batch_img, batch_att, batch_label) in enumerate(train_loader):
        batch_img = batch_img.to(device)
        batch_att = batch_att.to(device)
        batch_label = batch_label.to(device)

        pred_attri, pred_class = net(batch_img, att_seen)
        loss_ce = loss_cls(pred_class, batch_label)
        loss_mse = loss_reg(pred_attri, batch_att)
        loss = loss_ce + loss_mse

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_epoch.append(loss.item())

    losses += loss_epoch
    losses_mean = sum(losses) / len(losses)

    log_info = 'epoch: %d | loss: %.4f, loss cls: %.4f, loss reg: %.4f lr: %.6f' % \
               (epoch + 1, losses_mean, loss_ce.item(), loss_mse.item(), opt.param_groups[0]["lr"])
    print(log_info)

    with torch.no_grad():
        bias_s = torch.zeros((1, 150)).fill_(0.7).to(device)
        bias_u = torch.zeros((1, 50)).to(device)
        bias = torch.cat([bias_s, bias_u], dim=1)

        train_acc = cal_acc(train_loader, net, att_seen, train_id, bias=None)
        acc_seen = cal_acc(seen_loader, net, at_map, train_text_id, bias=bias)
        acc_unseen = cal_acc(unseen_loader, net, at_map, train_text_id, bias=bias)
    H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
    if best_gzsl_acc < H:
        best_acc_seen, best_acc_unseen, best_gzsl_acc = acc_seen, acc_unseen, H
    print('trian acc: %.4f, gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (train_acc, acc_seen, acc_unseen, H))
print('the best GZSL seen accuracy is %.4f' % best_acc_seen)
print('the best GZSL unseen accuracy is %.4f' % best_acc_unseen)
print('the best GZSL H is %.4f' % best_gzsl_acc)
