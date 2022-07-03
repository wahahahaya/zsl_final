import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from zmq import device


class CLASSIFIER:
    def __init__(self, in_dim, nclass, synth_loader, train_loader, seen_loader, unseen_loader, device):
        self.device = device
        self.nepoch = 20
        self.in_dim = in_dim
        self.nclass = nclass
        self.synloader = synth_loader
        self.trloader = train_loader
        self.seenloader = seen_loader
        self.unseenloader = unseen_loader

        self.model = LINEAR_LOGSOFTMAX_CLASSIFIER(self.in_dim, self.nclass)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.5, 0.999))

        self.model.to(self.device)
        self.criterion.to(self.device)

        self.acc_seen, self.acc_unseen, self.H, self.train_acc = self.fit()

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        best_train = 0
        for epoch in range(self.nepoch):
            for iteration, (feature, label) in enumerate(self.synloader):
                self.model.zero_grad()
                feature = feature.to(self.device)
                label = label.to(self.device)

                pred = self.model(feature)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optimizer.step()

            acc_seen = 0
            acc_unseen = 0
            acc_train = 0
            with torch.no_grad():
                acc_seen = self.cal_acc(self.seenloader)
                acc_unseen = self.cal_acc(self.unseenloader)
                acc_train = self.cal_acc(self.trloader)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)

        if H > best_H:
            best_seen = acc_seen
            best_unseen = acc_unseen
            best_H = H
            best_train = acc_train
        return best_seen, best_unseen, best_H, best_train

    def cal_acc(self, loader):
        scores = []
        labels = []
        for iteration, (feature, label) in enumerate(loader):
            feature = feature.to(device)
            label = label.to(device)

            pred_class = self.model(feature)
            scores.append(pred_class)
            labels.append(label)

        scores = torch.cat(scores, dim=0)
        labels = torch.cat(labels, dim=0)
        _, pred = scores.max(dim=1)
        outpred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
        unique_labels = np.unique(labels)

        acc = 0
        for i in unique_labels:
            idx = np.nonzero(labels == i)[0]
            acc += accuracy_score(labels[idx], outpred[idx])
        acc = acc / unique_labels.shape[0]

        return acc


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
