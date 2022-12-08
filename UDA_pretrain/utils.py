"""
@author: Junguang Jiang, Baixu Chen
@contact: JiangJunguang1123@outlook.com, cbx_99_hasta@outlook.com
"""
import sys
import os.path as osp
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import ConcatDataset
import wilds
import torchattacks
import foolbox as fb
from einops import rearrange
from RMA import *

sys.path.append('../../..')
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
import pylab
import matplotlib.pyplot as plt


def get_model_names():
    return sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    ) + timm.list_models()


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

def attack_pgd(X, y, model, attack_iters=10, step_size=2 / 255., epsilon=8 / 255.):
    was_training = model.training
    model.eval()
    # preprocessing = dict(mean=cifar10_mean, std=cifar10_std, axis=-3)

    fmodel = fb.models.PyTorchModel(model, bounds=(0, 1))
    pgd_attack = fb.attacks.PGD(abs_stepsize=step_size, steps=attack_iters, random_start=True)

    raw, clipped, is_adv = pgd_attack(fmodel, X, y, epsilons=epsilon)

    model.train(was_training)
    return clipped

def get_dataset_names():
    return sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    ) + wilds.supported_datasets + ['Digits']


def get_dataset(dataset_name, root, source, target, train_source_transform, val_transform, train_target_transform=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    if dataset_name == "Digits":
        train_source_dataset = datasets.__dict__[source[0]](osp.join(root, source[0]), download=True,
                                                            transform=train_source_transform)
        train_target_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), download=True,
                                                            transform=train_target_transform)
        val_dataset = test_dataset = datasets.__dict__[target[0]](osp.join(root, target[0]), split='test',
                                                                  download=True, transform=val_transform)
        class_names = datasets.MNIST.get_classes()
        num_classes = len(class_names)
    elif dataset_name in datasets.__dict__:
        # load datasets from common.vision.datasets
        dataset = datasets.__dict__[dataset_name]

        def concat_dataset(tasks, **kwargs):
            return ConcatDataset([dataset(task=task, **kwargs) for task in tasks])

        train_source_dataset = concat_dataset(root=root, tasks=source, download=True, transform=train_source_transform)
        train_target_dataset = concat_dataset(root=root, tasks=target, download=True, transform=train_target_transform)
        val_dataset = concat_dataset(root=root, tasks=target, download=True, transform=val_transform)
        if dataset_name == 'DomainNet':
            test_dataset = concat_dataset(root=root, tasks=target, split='test', download=True, transform=val_transform)
        else:
            test_dataset = val_dataset
        class_names = train_source_dataset.datasets[0].classes
        num_classes = len(class_names)
    else:
        # load datasets from wilds
        dataset = wilds.get_dataset(dataset_name, root_dir=root, download=True)
        num_classes = dataset.n_classes
        class_names = None
        train_source_dataset = convert_from_wilds_dataset(dataset.get_subset('train', transform=train_source_transform))
        train_target_dataset = convert_from_wilds_dataset(dataset.get_subset('test', transform=train_target_transform))
        val_dataset = test_dataset = convert_from_wilds_dataset(dataset.get_subset('test', transform=val_transform))
    return train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, class_names

def get_attack(model, steps=20, eps=8/255, alpha=2/255):
    return torchattacks.PGD(model=model,
                            eps=eps,
                            alpha=alpha,
                            steps=steps,
                            random_start=True)
def validate(val_loader, model, args, device) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_adv = AverageMeter('Loss_adv', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top1_adv = AverageMeter('Acc@1_adv', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1,losses_adv,top1_adv],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    attack_method = get_attack(model, steps=20, eps=args.eps, alpha=args.alpha)
    if args.per_class_eval:
        confmat = ConfusionMatrix(len(args.class_names))
    else:
        confmat = None

    
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        images_adv = attack_method(images, target)
        with torch.no_grad():
            # compute output
            output = model(images)
            output_adv=model(images_adv)
            loss = F.cross_entropy(output, target)
            loss_adv = F.cross_entropy(output_adv, target)
            
            # measure accuracy and record loss
            acc1, = accuracy(output, target, topk=(1,))
            acc1_adv, = accuracy(output_adv, target, topk=(1,))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            losses_adv.update(loss_adv.item(), images.size(0))
            top1_adv.update(acc1_adv.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    print(' * Acc@1 {top1.avg:.3f},Acc@1_adv {top1_adv.avg:.3f}'.format(top1=top1,top1_adv=top1_adv))
    if confmat:
        print(confmat.format(args.class_names))

    return top1.avg,top1_adv.avg

class get_train_transform(object):
    def __init__(self, args,resizing='default', random_horizontal_flip=True, random_color_jitter=False,
                 resize_size=224, norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    
        """
        resizing mode:
            - default: resize the image to 256 and take a random resized crop of size 224;
            - cen.crop: resize the image to 256 and take the center crop of size 224;
            - res: resize the image to 224;
        """
        args.window_size = (resize_size // args.patch_size, resize_size // args.patch_size)
        if resizing == 'default':
            transform = T.Compose([
                ResizeImage(256),
                T.RandomResizedCrop(224)
            ])
        elif resizing == 'cen.crop':
            transform = T.Compose([
                ResizeImage(256),
                T.CenterCrop(224)
            ])
        elif resizing == 'ran.crop':
            transform = T.Compose([
                ResizeImage(256),
                T.RandomCrop(224)
            ])
        elif resizing == 'res.':
            transform = ResizeImage(resize_size)
        else:
            raise NotImplementedError(resizing)
        self.transforms = [transform]
        self.masked_position_generator = RandomMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        if random_horizontal_flip:
            self.transforms.append(T.RandomHorizontalFlip())
        if random_color_jitter:
            self.transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
        self.transforms.extend([
            T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
        ])
        self.transforms=T.Compose(self.transforms)
    
    def __call__(self, image):
        return self.transforms(image), self.masked_position_generator()

def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    """
    resizing mode:
        - default: resize the image to 256 and take the center crop of size 224;
        – res.: resize the image to 224
    """
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

def pretrain(train_source_iter, model, optimizer, lr_scheduler, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)

        cls_loss = F.cross_entropy(y_s, labels_s)
        loss = cls_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

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

def imshow(img):
    fig = plt.figure(figsize=(10, 10))
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    pylab.show()
    fig.savefig('111.svg', format='svg', dpi=150)