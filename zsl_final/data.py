import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import scipy.io as sio
from os.path import join
from PIL import Image


class DATA_LOADER(object):
    def __init__(self, dataname):
        self.data_name = dataname
        self.read_matdataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0

        self.train_loader = self.loader(self.train_image_file, self.train_attri, self.train_label)
        self.seen_loader = self.loader(self.seen_image_file, self.seen_attri, self.seen_label)
        self.unseen_loader = self.loader(self.unseen_image_file, self.unseen_attri, self.unseen_label)

    def read_matdataset(self):
        matcontent = sio.loadmat("./data/"+self.data_name+"/res101.mat")
        # res101.mat
        img_files = matcontent['image_files'].squeeze()
        new_img_files = []
        for img_file in img_files:
            img_path = img_file[0]
            if self.data_name == 'CUB':
                img_path = join("../dataset/CUB/CUB_200_2011/", '/'.join(img_path.split('/')[7:]))
            new_img_files.append(img_path)

        self.image_files = np.array(new_img_files)
        label = matcontent['labels'].astype(int).squeeze() - 1

        matcontent = sio.loadmat("./data/"+self.data_name+"/att_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        # train
        self.train_image_file = self.image_files[trainval_loc]
        train_idx = label[trainval_loc].astype(int)
        self.train_attri = self.attribute[train_idx]  # CUB: (7057, 312)
        train_id, train_index = np.unique(train_idx, return_inverse=True)
        self.train_label = torch.from_numpy(train_index).long()

        # seen
        self.seen_image_file = self.image_files[test_seen_loc]
        seen_idx = label[test_seen_loc].astype(int)
        self.seen_attri = self.attribute[seen_idx]
        seen_id, seen_index = np.unique(seen_idx, return_inverse=True)
        self.seen_label = torch.from_numpy(seen_index).long()

        # unseen
        num_train = len(train_id)
        self.unseen_image_file = self.image_files[test_unseen_loc]
        unseen_idx = label[test_unseen_loc].astype(int)
        self.unseen_attri = self.attribute[unseen_idx]
        unseen_id, unseen_index = np.unique(unseen_idx, return_inverse=True)
        self.unseen_label = torch.from_numpy(unseen_index+num_train).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.unseen_label.numpy()))

        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.attribute_seen = self.attribute[seen_id]
        self.attribute_unseen = self.attribute[unseen_id]

        self.train_id = np.unique(self.train_label)
        self.test_id = np.unique(self.unseen_label)
        self.train_test_id = np.concatenate((self.train_id, self.test_id))

    def loader(self, image, attribute, label):
        transforms = data_transform('resize_random_crop', size=224)
        dataset = RandDataset(image, attribute, label, transforms)
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=64, drop_last=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=8,
            batch_sampler=batch_sampler,
        )

        return loader


def data_transform(name, size=224):
    name = name.strip().split('+')
    name = [n.strip() for n in name]
    transform = []

    if 'resize_random_crop' in name:
        transform.extend([
            transforms.Resize(int(size * 8. / 7.)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(0.5)
        ])
    elif 'resize_center_crop' in name:
        transform.extend(
            transforms.Resize(size),
            transforms.CenterCrop(size),
        )
    elif 'resize_only' in name:
        transform.extend([
            transforms.Resize((size, size)),
        ])
    elif 'resize' in name:
        transform.extend([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(0.5)
        ])
    else:
        transform.extend([
            transforms.Resize(size),
            transforms.CenterCrop(size)
        ])
    if 'colorjitter' in name:
        transform.extend(
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2)
        )
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform)
    return transform


class RandDataset(data.Dataset):
    def __init__(self, img_path, atts, labels, transforms=None):
        self.img_path = img_path
        self.atts = atts.clone().detach().float()
        self.labels = labels.clone().detach().long()
        self.classes = np.unique(labels)

        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        label = self.labels[index]
        att = self.atts[index]

        return img, att, label

    def __len__(self):
        return self.labels.size(0)
