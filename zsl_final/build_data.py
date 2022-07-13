from os.path import join
from scipy import io
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.utils.data as data

from utils import seed_worker


class ImgDatasetParam(object):
    DATASETS = {
        "imgroot": "/HDD-1_data/arlen/dataset",
        "dataroot": "datasets/Data",
        "image_embedding": "res101",
        "class_embedding": "att"
    }

    @staticmethod
    def get(dataset):
        attrs = ImgDatasetParam.DATASETS
        # attrs["imgroot"] = join(attrs["imgroot"], dataset)
        args = dict(
            dataset=dataset
        )
        args.update(attrs)
        return args


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


class CategoriesSampler():
    def __init__(self, label_for_imgs, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch  # batchs for each epoch
        self.n_cls = n_cls  # ways
        self.n_per = n_per  # shots
        self.ep_per_batch = ep_per_batch  # episodes for each batch, defult set 1

        self.cat = list(np.unique(label_for_imgs))
        self.catlocs = {}

        for c in self.cat:
            self.catlocs[c] = np.argwhere(label_for_imgs == c).reshape(-1)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(self.cat, self.n_cls, replace=False)

                for c in selected_classes:
                    ll = np.random.choice(self.catlocs[c], self.n_per, replace=False)
                    episode.append(torch.from_numpy(ll))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch)  # bs * n_cls * n_per
            yield batch.view(-1)


class ZSLDataset(data.Dataset):
    def __init__(self, img_path, atts, labels, transforms=None):
        self.img_path = img_path
        self.atts = torch.tensor(atts).float()
        self.labels = labels.long()

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


def build_dataloader(config):
    args = ImgDatasetParam.get(config.dataset_name)
    imgroot = args['imgroot']
    dataroot = args['dataroot']
    image_embedding = args['image_embedding']
    class_embedding = args['class_embedding']
    dataset = args['dataset']

    img_mat = io.loadmat(dataroot+'/'+dataset+'/'+image_embedding+'.mat')
    img_files = np.squeeze(img_mat['image_files'])
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]
        if dataset == "CUB":
            img_path = join(imgroot+'/CUB', '/'.join(img_path.split('/')[6:]))
        elif dataset == "AWA2":
            img_path = join(imgroot+'/AWA2', '/'.join(img_path.split('/')[5:]))
        elif dataset == "SUN":
            img_path = join(imgroot+'/SUN', '/'.join(img_path.split('/')[7:]))
        new_img_files.append(img_path)

    new_img_files = np.array(new_img_files)
    # ex. 1~50 --> 0~49
    label = img_mat['labels'].astype(int).squeeze() - 1

    att_mat = io.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
    trainvalloc = att_mat['trainval_loc'].squeeze() - 1
    test_seen_loc = att_mat['test_seen_loc'].squeeze() - 1
    test_unseen_loc = att_mat['test_unseen_loc'].squeeze() - 1
    cls_name = att_mat['allclasses_names']
    attribute = att_mat['att'].T

    # training set
    train_img = new_img_files[trainvalloc]
    train_label = label[trainvalloc]
    train_att = attribute[train_label]

    train_id, idx = np.unique(train_label, return_inverse=True)
    train_att_unique = attribute[train_id]
    train_clsname = cls_name[train_id]

    num_train = len(train_id)
    train_label = idx
    train_id = np.unique(train_label)

    # testing unseen set
    test_img_unseen = new_img_files[test_unseen_loc]
    test_label_unseen = label[test_unseen_loc].astype(int)
    unseen_att = attribute[test_label_unseen]
    test_id, idx = np.unique(test_label_unseen, return_inverse=True)
    att_unseen = attribute[test_id]
    test_clsname = cls_name[test_id]
    test_label_unseen = idx + num_train
    test_id = np.unique(test_label_unseen)

    train_test_att = np.concatenate((train_att_unique, att_unseen))
    train_test_id = np.concatenate((train_id, test_id))

    test_img_seen = new_img_files[test_seen_loc]
    test_label_seen = label[test_seen_loc].astype(int)
    seen_att = attribute[test_label_seen]
    _, idx = np.unique(test_label_seen, return_inverse=True)
    test_label_seen = idx

    att_unseen = torch.from_numpy(att_unseen).float()
    test_label_seen = torch.tensor(test_label_seen)
    test_label_unseen = torch.tensor(test_label_unseen)
    train_label = torch.tensor(train_label)
    att_seen = torch.from_numpy(train_att_unique).float()

    res = {
        'train_label': train_label,
        'train_att': train_att,
        'test_label_seen': test_label_seen,
        'test_label_unseen': test_label_unseen,
        'att_unseen': att_unseen,
        'att_seen': att_seen,
        'train_id': train_id,
        'test_id': test_id,
        'train_test_id': train_test_id,
        'train_clsname': train_clsname,
        'test_clsname': test_clsname
    }

    ways = config.ways
    shots = config.shots
    img_size = config.image_size
    train_aug = config.train_aug
    test_aug = config.test_aug

    train_transforms = data_transform(train_aug, size=img_size)
    n_batch = config.n_batch
    train_dataset = ZSLDataset(train_img, train_att, train_label, train_transforms)
    sampler = CategoriesSampler(train_label, n_batch, ways, shots)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=sampler, num_workers=23, pin_memory=True, worker_init_fn=seed_worker)

    test_transforms = data_transform(test_aug, size=img_size)
    test_batch = config.test_batch
    test_unseen_dataset = ZSLDataset(test_img_unseen, unseen_att, test_label_unseen, test_transforms)
    test_unseen_dataloader = DataLoader(
        test_unseen_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=23,
        pin_memory=False,
        worker_init_fn=seed_worker
    )

    test_seen_dataset = ZSLDataset(test_img_seen, seen_att, test_label_seen, test_transforms)
    test_seen_dataloader = DataLoader(
        test_seen_dataset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=23,
        pin_memory=False,
        worker_init_fn=seed_worker
    )

    return train_dataloader, test_seen_dataloader, test_unseen_dataloader, res
