# code in this file is adpated from
# https://github.com/kekmodel/MPL-pytorch/blob/main/main.py

import operator
import math
import os
import torchvision.utils as vutils
import torchvision

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import torchattacks
import random
from torch.cuda import amp
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from dalib.adaptation.mdd import ClassificationMarginDisparityDiscrepancy \
    as MarginDisparityDiscrepancy, ImageClassifier
from utils import *
from RMA import *

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))

        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_attack(model, steps=10, eps=8 / 255, alpha=2 / 255):
    return torchattacks.PGD(model=model,
                            eps=eps,
                            alpha=alpha,
                            steps=steps,
                            random_start=True)


def train_loop(args, source_loader, target_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.name}")
    logger.info(f"   Total steps = {args.total_steps}")
    best_acc = 0
    if args.world_size > 1:
        source_epoch = 0
        target_epoch = 0
        source_loader.sampler.set_epoch(source_epoch)
        target_loader.sampler.set_epoch(target_epoch)

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mate = AverageMeter()
            mean_mask = AverageMeter()
            acc = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()
        try:
            images_l, targets = source_iter.next()

        except:
            if args.world_size > 1:
                source_epoch += 1
                source_loader.sampler.set_epoch(source_epoch)
            source_iter = iter(source_loader)
            images_l, targets = source_iter.next()
        try:
            (images_uw, images_us), tu = target_iter.next()
        #            images_us = get_Mask_data(args, images_uw ,images_us)

        except:
            if args.world_size > 1:
                target_epoch += 1
                target_loader.sampler.set_epoch(target_epoch)
            target_iter = iter(target_loader)
            (images_uw, images_us), tu = target_iter.next()
        #            images_us = get_Mask_data(args, images_uw, images_us)

        tu = [int(i) for i in tu]
        tu = torch.tensor(np.array(tu))
        images_l = images_l.to(args.device)
        images_uw = images_uw.to(args.device)
        images_us = images_us.to(args.device)
        tu = tu.to(args.device)
        targets = targets.to(args.device)

        targets = targets.to(torch.int64)

        with amp.autocast(enabled=args.amp):
            batch_size = images_l.shape[0]
            t_images = torch.cat((images_l, images_uw, images_us))
            t_logits, _ = teacher_model(t_images)
            t_logits_l = t_logits[:batch_size]
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
            del t_logits

            t_loss_l = criterion(t_logits_l, targets)

            soft_pseudo_label = torch.softmax(t_logits_uw.detach() / args.temperature, dim=-1)
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u
            attack_method = get_attack(student_model, eps=args.eps, alpha=args.alpha)
            images_us_adv = attack_method(images_us, hard_pseudo_label.detach())

            s_images = torch.cat((images_l, images_us_adv))

            s_logits, _ = student_model(s_images)

            s_logits_l = s_logits[:batch_size]

            n = 9 * batch_size
            s_logits_us = s_logits[batch_size:]
            del s_logits

            pseudo_labeling_acc = (hard_pseudo_label == tu) * 1
            pseudo_labeling_acc = (sum(pseudo_labeling_acc) / len(pseudo_labeling_acc)) * 100

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label.detach())

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if args.ema > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l, _ = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

            # theoretically correct formula (https://github.com/kekmodel/MPL-pytorch/issues/6)
            # dot_product = s_loss_l_old - s_loss_l_new

            # author's code formula
            dot_product = s_loss_l_new - s_loss_l_old
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product

            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mate = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            # test
            # t_loss_mate = torch.tensor(0.).to(args.device)
            t_loss = t_loss_uda + t_loss_mate
            # t_loss = t_loss.detach_().requires_grad_(True)

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()
        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mate = reduce_tensor(t_loss_mate.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mate.update(t_loss_mate.item())
        mean_mask.update(mask.mean().item())
        acc.update(pseudo_labeling_acc)

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. pseudo_labeling_acc: {acc.avg:.4f} ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            # wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mate", t_losses_mate.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                args.writer.add_scalar("train/7.pseudo acc", acc.avg, args.num_eval)
                # wandb.log({"train/1.s_loss": s_losses.avg,
                #            "train/2.t_loss": t_losses.avg,
                #            "train/3.t_labeled": t_losses_l.avg,
                #            "train/4.t_unlabeled": t_losses_u.avg,
                #            "train/5.t_mate": t_losses_mate.avg,
                #            "train/6.mask": mean_mask.avg})
                test_Tmodel = teacher_model
                test_Tmodel.eval()
                test_model = avg_student_model if avg_student_model is not None else student_model
                test_model.eval()
                test_loss, top1, top5, top1_adv, top5_adv, pseudo_acc = evaluate(args, test_loader, test_model,
                                                                                 test_Tmodel, criterion)
                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc@1", top1, args.num_eval)
                args.writer.add_scalar("test/acc@5", top5, args.num_eval)
                args.writer.add_scalar("test/acc@1_adv", top1_adv, args.num_eval)
                args.writer.add_scalar("test/acc@5_adv", top5_adv, args.num_eval)
                args.writer.add_scalar("test/acc_pseudo", pseudo_acc, args.num_eval)
                # wandb.log({"test/loss": test_loss,
                #            "test/acc@1": top1,
                #            "test/acc@5": top5})
                is_best = top1_adv >= args.best_top1

                if is_best:
                    args.best_top1 = top1_adv
                    args.best_top5 = top5_adv
                    args.best_acc = top1

                logger.info(f"top-1 acc: {top1_adv:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")
                logger.info(f"pseudo acc: {pseudo_acc:.2f}")
                logger.info(f"Best acc: {args.best_acc:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'best_acc': args.best_acc,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if args.local_rank in [-1, 0]:
        args.writer.add_scalar("result/test_acc@1", args.best_top1)
    return


def evaluate(args, test_loader, model, Tmodel, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    pseudo_acc = AverageMeter()
    losses_adv = AverageMeter()
    top1_adv = AverageMeter()
    top5_adv = AverageMeter()
    model.eval()
    Tmodel.eval()
    attack_method = get_attack(model, steps=20, eps=args.eps, alpha=args.alpha)
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])

    end = time.time()

    for step, (images, targets) in enumerate(test_iter):
        data_time.update(time.time() - end)
        batch_size = images.shape[0]
        targets = [int(i) for i in targets]
        targets = torch.tensor(np.array(targets))
        images = images.to(args.device)
        targets = targets.to(args.device)
        targets = targets.to(torch.int64)

        #            with torch.no_grad():
        images_adv = attack_method(images, targets)
        #        images_adv=attack_pgd(images,targets, model, attack_iters=20)
        with torch.no_grad():
            with amp.autocast(enabled=args.amp):
                outputs = model(images)
                outputs_adv = model(images_adv)
                t_logits = Tmodel(images)
                soft_pseudo_label = torch.softmax(t_logits.detach() / args.temperature, dim=-1)
                max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
                loss = criterion(outputs, targets)
                loss_adv = criterion(outputs_adv, targets)
                pseudo_labeling_acc = (hard_pseudo_label == targets) * 1
                pseudo_labeling_acc = (sum(pseudo_labeling_acc) / len(pseudo_labeling_acc)) * 100

            acc1, acc5 = accuracy(outputs, targets, (1, 5))
            acc1_adv, acc5_adv = accuracy(outputs_adv, targets, (1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            pseudo_acc.update(pseudo_labeling_acc)
            losses_adv.update(loss_adv.item(), batch_size)
            top1_adv.update(acc1_adv[0], batch_size)
            top5_adv.update(acc5_adv[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step + 1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}.Loss_adv: {losses_adv.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}.top1_adv: {top1_adv.avg:.2f}. top5_adv: {top5_adv.avg:.2f}.pseudo_acc: {pseudo_acc.avg:.2f}. ")

    test_iter.close()
    return losses.avg, top1.avg, top5.avg, top1_adv.avg, top5_adv.avg, pseudo_acc.avg


def SRoUDA_Mate_step(args, train_source_dataset, train_target_dataset, val_dataset, num_classes):
    # args = parser.parse_args()
    args.best_top1 = 0.
    args.best_top5 = 0.
    args.best_acc = 0

    if args.local_rank != -1:
        args.gpu = args.local_rank
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
    else:
        args.gpu = 0
        args.world_size = 1

    args.device = torch.device('cuda', args.gpu)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")

    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.local_rank == 0:
        torch.distributed.barrier()

    if True:  # args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = len(train_target_dataset) // args.Matebatch_size * args.mu // num_tasks

        sampler_train = torch.utils.data.DistributedSampler(
            train_target_dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    source_loader = DataLoader(
        train_source_dataset,
        sampler=train_sampler(train_source_dataset),
        batch_size=args.Matebatch_size,
        num_workers=args.workers,
        drop_last=True)

    target_loader = DataLoader(
        train_target_dataset,
        sampler=train_sampler(train_target_dataset),
        batch_size=args.Matebatch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)

    #    target_loader = DataLoader(
    #        train_target_dataset, sampler=sampler_train,
    #        batch_size=args.Matebatch_size * args.mu,
    #        num_workers=args.workers,
    #        drop_last=True,
    #        worker_init_fn=seed_worker
    #    )

    test_loader = DataLoader(val_dataset,
                             sampler=SequentialSampler(val_dataset),
                             batch_size=64,
                             num_workers=args.workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    print("=> using model '{}'".format(args.arch))

    backbone = get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    teacher_model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                    width=args.bottleneck_dim, pool_layer=pool_layer)

    backbone = get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    student_model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                    width=args.bottleneck_dim, pool_layer=pool_layer)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    # logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")
    print(args.model_path)
    teacher_model.to(args.device)
    teacher_model.load_state_dict(torch.load(args.model_path))
    student_model.to(args.device)
    student_model.load_state_dict(torch.load(args.model_path))
    avg_student_model = None
    # if args.ema > 0:
    #     avg_student_model = ModelEMA(student_model, args.ema)

    #    criterion = create_loss_fn(args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    no_decay = ['bn']

    teacher_parameters = [
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)]},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)]}
    ]
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(
            nd in n for nd in no_decay)]},
        {'params': [p for n, p in student_model.named_parameters() if any(
            nd in n for nd in no_decay)]}
    ]
    t_optimizer = optim.SGD(teacher_parameters,
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            if not args.evaluate:
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(args, source_loader, target_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler)
    return
