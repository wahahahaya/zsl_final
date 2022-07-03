from os.path import join
import torch
import numpy as np
from scipy import io
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


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


def build_dataloader(is_distributed=False):
    imgroot = "../../dataset/CUB/CUB_200_2011/"
    dataroot = "./data/"
    image_embedding = "res101"
    class_embedding = "att_splits"
    dataset = 'CUB'

    matcontent = io.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")

    img_files = np.squeeze(matcontent['image_files'])
    new_img_files = []
    for img_file in img_files:
        img_path = img_file[0]
        img_path = join(imgroot, '/'.join(img_path.split('/')[7:]))
        new_img_files.append(img_path)

    new_img_files = np.array(new_img_files)
    label = matcontent['labels'].astype(int).squeeze() - 1

    matcontent = io.loadmat(dataroot + "/" + dataset + "/" + class_embedding + ".mat")
    trainvalloc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

    att_name = 'att'
    # if dataset == 'AwA2':
    #     att_name = 'original_att'
    cls_name = matcontent['allclasses_names']

    attribute = matcontent[att_name].T

    train_img = new_img_files[trainvalloc]
    train_label = label[trainvalloc].astype(int)
    train_att = attribute[train_label]


    train_id, idx = np.unique(train_label, return_inverse=True)
    train_att_unique = attribute[train_id]
    train_clsname = cls_name[train_id]

    num_train = len(train_id)
    train_label = idx
    train_id = np.unique(train_label)

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
        'seen_att': seen_att,
        'unseen_att': unseen_att,
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

    # train dataloader
    data_aug_train = "resize_random_crop"
    img_size = 224
    transforms = data_transform(data_aug_train, size=img_size)
    batch = 64

    dataset = RandDataset(train_img, train_att, train_label, transforms)
    sampler = torch.utils.data.sampler.RandomSampler(dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch, drop_last=True)
    tr_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=8,
        batch_sampler=batch_sampler,
    )

    data_aug_test = "resize_crop"
    transforms = data_transform(data_aug_test, size=img_size)
    test_batch_size = 100

    if not is_distributed:
        # test unseen dataloader
        tu_data = RandDataset(test_img_unseen, unseen_att, test_label_unseen, transforms)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)

        # test seen dataloader
        ts_data = RandDataset(test_img_seen, seen_att, test_label_seen, transforms)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, shuffle=False,
            num_workers=4, pin_memory=False)
    else:
        # test unseen dataloader
        tu_data = RandDataset(test_img_unseen, unseen_att, test_label_unseen, transforms)
        tu_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tu_data, shuffle=False)
        tu_loader = torch.utils.data.DataLoader(
            tu_data, batch_size=test_batch_size, sampler=tu_sampler,
            num_workers=4, pin_memory=False)

        # test seen dataloader
        ts_data = RandDataset(test_img_seen, seen_att, test_label_seen, transforms)
        ts_sampler = torch.utils.data.distributed.DistributedSampler(dataset=ts_data, shuffle=False)
        ts_loader = torch.utils.data.DataLoader(
            ts_data, batch_size=test_batch_size, sampler=ts_sampler,
            num_workers=4, pin_memory=False)

    return tr_dataloader, tu_loader, ts_loader, res


class TestDataset(data.Dataset):

    def __init__(self, img_path, labels, transforms=None):
        self.img_path = img_path
        self.labels = torch.tensor(labels).long()
        self.classes = np.unique(labels)

        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        label = self.labels[index]

        return img, label

    def __len__(self):
        return self.labels.size(0)


class RandDataset(data.Dataset):

    def __init__(self, img_path, atts, labels, transforms=None):
        self.img_path = img_path
        self.atts = torch.tensor(atts).float()
        self.labels = torch.tensor(labels).long()
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
