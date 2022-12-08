import logging
import os

import torchattacks
import wilds
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import shutil
import os.path as osp
import time
import torch.nn.functional as F
from typing import Optional, List
from torch import distributed as dist

import timm
from einops import rearrange

from torchvision.datasets.folder import default_loader
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import datasets, transforms
from functools import partial
from collections import OrderedDict

import common.vision.models as models
from common.vision.datasets.officehome import OfficeHome
from common.vision.datasets.visda2017 import VisDA2017
from common.vision.transforms import ResizeImage
from common.vision.datasets.office31 import Office31
import common.vision.datasets as dataset
from RMA import RandomMaskingGenerator
import matplotlib.pyplot as plt
import numpy as np
import pylab
import ssl
import prettytable
from augmentation import RandAugmentCIFAR,RandAugmentUSPS,RandAugmentSVHN,RandAugmentOFFICE

ssl._create_default_https_context = ssl._create_unverified_context


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    pylab.show()

def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def init_logger(out_dir, filename=None, level=logging.INFO):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if filename is not None:
        logfile = os.path.join(out_dir, filename)
    else:
        logfile = os.path.join(out_dir, 'log.txt')

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=level,
        filename=logfile)

    return logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(args, state, is_best, finetune=False):
    os.makedirs(args.log, exist_ok=True)
    name = args.name
    filename = f'{args.log}/last.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{args.log}/best.pth.tar')

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        """
        Update confusion matrix.

        Args:
            target: ground truth
            output: predictions of models

        Shape:
            - target: :math:`(minibatch, C)` where C means the number of classes.
            - output: :math:`(minibatch, C)` where C means the number of classes.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """compute global accuracy, per-class accuracy and per-class IoU"""
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

    def format(self, classes: list):
        """Get the accuracy and IoU for each class in the table format"""
        acc_global, acc, iu = self.compute()

        table = prettytable.PrettyTable(["class", "acc", "iou"])
        for i, class_name, per_acc, per_iu in zip(range(len(classes)), classes, (acc * 100).tolist(), (iu * 100).tolist()):
            table.add_row([class_name, per_acc, per_iu])

        return 'global correct: {:.1f}\nmean correct:{:.1f}\nmean IoU: {:.1f}\n{}'.format(
            acc_global.item() * 100, acc.mean().item() * 100, iu.mean().item() * 100, table.get_string())

def get_model(model_name, pretrain=True):
    if model_name in models.__dict__:
        # load models from common.vision.models
        backbone = models.__dict__[model_name](pretrained=pretrain)
    else:
        # load models from pytorch-image-models
        backbone = timm.create_model(model_name, pretrained=pretrain)
        try:
            backbone.out_features = backbone.get_classifier().in_features
            backbone.reset_classifier(0, '')
        except:
            backbone.out_features = backbone.head.in_features
            backbone.head = nn.Identity()
    return backbone

def create_loss_fn(args):
    # if args.label_smoothing > 0:
    #     criterion = SmoothCrossEntropyV2(alpha=args.label_smoothing)
    # else:
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    return criterion.to(args.device)


def module_load_state_dict(model, state_dict):
    try:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = f'module.{k}'  # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def model_load_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except:
        module_load_state_dict(model, state_dict)

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()

def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + wilds.supported_datasets + ['Digits']

def get_attack(model, steps=20):
    return torchattacks.PGD(model=model,
                            eps=8 / 255,
                            alpha=2 / 255,
                            steps=steps,
                            random_start=True)

class TransformAUG(object):
    def __init__(self, args,resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                 resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
        args.window_size = (resize_size // args.patch_size, resize_size // args.patch_size)
        if args.randaug:
            n, m = args.randaug
        else:
            n, m = 2, 10  # default
        if resizing == 'default':
            transform = transforms.Compose([
                        ResizeImage(256),
                        transforms.CenterCrop(224),
                    ])
        elif resizing == 'res.':
            transform = ResizeImage(resize_size)
        else:
             raise NotImplementedError(resizing)

        self.ori = transforms.Compose([
            transform
        ])
        self.aug = transforms.Compose([
            transform,
            RandAugmentUSPS(n=n, m=m)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
#            transforms.Normalize(mean=norm_mean, std=norm_std)
         ])
        self.unnormalize = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, x):
        ori = self.ori(x)
        aug = self.aug(x)
        return self.normalize(ori), self.normalize(aug)

def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                        resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take a random resized crop of size 224;
        - cen.crop: resize the image to 256 and take the center crop of size 224;
        - res: resize the image to 224;
    """
    if resizing == 'default':
        transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomResizedCrop(224)
        ])
    elif resizing == 'cen.crop':
        transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = transforms.Compose([
            ResizeImage(256),
            transforms.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    trans = [transform]
    if random_horizontal_flip:
        trans.append(transforms.RandomHorizontalFlip())
    if random_color_jitter:
        trans.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    trans.extend([
        transforms.ToTensor(),
#        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    return transforms.Compose(trans)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return transforms.Compose([
        transform,
        transforms.ToTensor(),
#        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None,aug_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
#        print(source, target)
        train_source_dataset = dataset.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                        transform=train_source_transform)
        train_target_dataset = dataset.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                        transform=train_target_transform)
        train_target_aug = dataset.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                    transform=aug_transform)
        val_dataset = test_dataset = dataset.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                              download=True, transform=val_transform)
        class_names = dataset.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in dataset.__dict__:
        # load datasets from common.vision.datasets
        # dataset = datasets.__dict__[dataset_name]
        if dataset_name=="Office31":
            data = Office31
        elif dataset_name=="Officehome":
            data = OfficeHome
        elif dataset_name == "VisDA2017":
            data = VisDA2017

        def concat_dataset(tasks, **kwargs):
            return ConcatDataset([data(task=task, **kwargs) for task in tasks])

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform)
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)

        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform)
        train_target_labels = train_target_dataset.datasets[0].get_target()
        train_labeled_idxs, train_unlabeled_idxs = test_set_split(train_target_labels, num_classes)

        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform,
                                              indexs=train_unlabeled_idxs)

        train_target_aug = concat_dataset(root=root, tasks=target, download=True, transform=aug_transform,
                                          indexs=train_unlabeled_idxs)
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform,
                                     indexs=train_labeled_idxs)
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform)
        else:
            test_dataset = val_dataset
    else:
        # load datasets from wilds
        data = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = data.n_classes
        class_names = None
        train_source_dataset = convert_from_wilds_dataset(data.get_subset('train', transform=train_source_transform))
        train_target_dataset = convert_from_wilds_dataset(data.get_subset('test', transform=train_target_transform))
        train_target_aug = convert_from_wilds_dataset(data.get_subset('test', transform=aug_transform))
        val_dataset = test_dataset = convert_from_wilds_dataset(data.get_subset('test', transform=val_transform))
    return train_source_dataset, train_target_dataset, train_target_aug, val_dataset, test_dataset, num_classes, class_names
def convert_from_wilds_dataset(wild_dataset):
    class Dataset:
        def __init__(self):
            self.dataset = wild_dataset

        def __getitem__(self, idx):
            x, y, metadata = self.dataset[idx]
            return x, y

        def __len__(self):
            return len(self.dataset)

    return Dataset()
def test_set_split(labels,num_classes):
    # label_per_class = args.num_labeled // args.num_classes
    num=len(labels)
    num_test=num/6
    test_per_class = int(num_test // num_classes)
    labels = np.array(labels)
    test_set_idx = []
    # unlabeled data: all training data
    train_set_idx = np.array(range(len(labels)))
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, test_per_class, False)
        test_set_idx.extend(idx)
    test_set_idx = np.array(test_set_idx)
    train_set_idx = np.array(list(set(train_set_idx).difference(set(test_set_idx))))
    np.random.shuffle(train_set_idx)
    return test_set_idx, train_set_idx
#
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     pylab.show()
#
def get_Mask_data(args, imgs ,mask_index):
    b, c, h, w = imgs.size()
    mask_index = mask_index.to(args.device, non_blocking=True).flatten(1).to(torch.bool)
    img_squeeze = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=args.patch_size,
                            p2=args.patch_size)
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
            img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    mask = torch.ones_like(img_patch)
    mask[mask_index] = 0
    mask = rearrange(mask, 'b n (p c) -> b n p c',c=c)
    h = int(h/args.patch_size)
    w = int(w/args.patch_size)
    mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=args.patch_size, p2=args.patch_size, h=h,
                     w=w)

    img_mask = imgs * mask
    return img_mask
