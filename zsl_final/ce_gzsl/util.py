import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import os
from logger import create_logger
import datetime


def initialize_exp(path, name):
    # """
    # Experiment initialization.
    # """
    # # dump parameters
    # params.dump_path = get_dump_path(params)
    # pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # create a logger
    time_stamp = datetime.datetime.now()

    time = time_stamp.strftime('%Y%m%d%H%M%S')

    logger = create_logger(os.path.join(path, name + '_' + time + '.log'))
    print('log_name:', name + '_' + time + '.log')
    # logger = create_logger(os.path.join(path, name +'.log'))
    logger.info('============ Initialized logger ============')
    # logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
    #                       in sorted(dict(vars(params)).items())))
    return logger


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename + '.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename + '.log', "a")
        f.write(message)
        f.close()


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imagenet':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(
            "/HDD-1_data/arlen/zsl_final/new_data/AWA2_feat.mat")
        feature = matcontent['features']
        matcontent = sio.loadmat(
            opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        self.all_file = matcontent['image_files']
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(
            opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()
        if not opt.validation:
            self.train_image_file = self.all_file[trainval_loc]
            self.test_seen_image_file = self.all_file[test_seen_loc]
            self.test_unseen_image_file = self.all_file[test_unseen_loc]

            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(
                    feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(
                    _test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(
                    label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(
                    _test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                self.test_seen_label = torch.from_numpy(
                    label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(
                    feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(
                    feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(
                    label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(
                    feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(
                    label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(
                feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(
                label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(
            np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(
            np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(
            0, self.ntrain_class + self.ntest_class).long()
        self.attribute_seen = self.attribute[self.seenclasses]

        # collect the data of each class

        self.train_samples_class_index = torch.tensor(
            [self.train_label.eq(i_class).sum().float() for i_class in self.train_class])
        #
        # import pdb
        # pdb.set_trace()

        # self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att
