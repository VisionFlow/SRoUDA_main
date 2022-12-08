import argparse
import utils
from Mate_selftrain_step import *

parser = argparse.ArgumentParser(description='DAN for Unsupervised Domain Adaptation')
    # dataset parameters
parser.add_argument('--root', default='data/office31', type=str, help='data path') ##
parser.add_argument('-d', '--data', metavar='DATA', default='office31')
parser.add_argument('-s', '--source', help='source domain(s)', nargs='+',default='A')##
parser.add_argument('-t', '--target', help='target domain(s)', nargs='+',default='W')##
parser.add_argument('--name', type=str, default='', help='experiment name')##
parser.add_argument('--train-resizing', type=str, default='default')##
parser.add_argument('--val-resizing', type=str, default='default')##
parser.add_argument('--resize-size', type=int, default=224, ##
                        help='the image size after resizing')
parser.add_argument('--no-hflip', action='store_true', 
                   help='no random horizontal flipping during training')
parser.add_argument('--norm-mean', type=float, nargs='+',##
                        default=(0.485, 0.456, 0.406), help='normalization mean')
parser.add_argument('--norm-std', type=float, nargs='+',##
                        default=(0.229, 0.224, 0.225), help='normalization std')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',##
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet50)')
parser.add_argument('--bottleneck-dim', default=1024, type=int,##
                        help='Dimension of bottleneck')
parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
parser.add_argument('--non-linear', default=False, action='store_true',##
                        help='whether not use the linear version')
parser.add_argument('--trade-off', default=1., type=float,##
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
parser.add_argument('--model-path', default='', type=str,  ##
                    help='The path of pre-trained source model')

parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--use-gpu', default=True, type=str, help='use gpu')
parser.add_argument('--class-names', default='', help='class-names')
parser.add_argument('-i', '--iters-per-epoch', default=2500, type=int, ##
                        help='Number of iterations per epoch')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
parser.add_argument('--seed', default='1', type=int,
                        help='seed for initializing training. ')
parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
parser.add_argument("--log", type=str, default='checkpoints/office_A2D_ours', ##
                        help="Where to save logs, checkpoints and debugging images.")

parser.add_argument('--num-steps', default=10, type=int,
                        help='perturb number of steps')
parser.add_argument('--total-steps', default=20000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=100, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=0, type=int, help='number of workers')
parser.add_argument('--num-classes', default=31, type=int, help='number of classes') ##
parser.add_argument('--resize', default=32, type=int, help='resize image') ##
parser.add_argument('--mask_ratio', default=0.25, type=float,
                        help='ratio of the visual tokens/patches need be masked')

parser.add_argument('--Matebatch-size', default=2, type=int, help='train batch size') ##
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default= 1e-7, type=float, help='train learning late')##
parser.add_argument('--student_lr', default=0.001, type=float, help='train learning late')##
parser.add_argument('--patch-size', default=8, type=int, help='train batch size') ##
# parser.add_argument('--input_size', default=224, type=int,
#                         help='images input size for backbone')

# parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
# parser.add_argument('--weight-decay', default=0, type=float, help='train weight decay')
parser.add_argument('--ema', default=0, type=float, help='EMA decay rate')
parser.add_argument('--warmup-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=1e9, type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')

parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=8, type=int, help='coefficient of unlabeled batch size')##
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss') ##
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("--eps", type=float, default=8/255)
parser.add_argument("--alpha", type=float, default=2/255)

def main():
    args = parser.parse_args()
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    TransformAUG = utils.TransformAUG(args, args.val_resizing, random_horizontal_flip=not args.no_hflip,
                                      random_color_jitter=False, resize_size=args.resize_size,
                                      norm_mean=args.norm_mean, norm_std=args.norm_std)
    # TransformMPL = utils.TransformMPL(args,mean=args.norm_mean, std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)
    print("TransformAUG: ", TransformAUG)

    train_source_dataset, train_target_dataset, train_target_RMA, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform,
                          aug_transform=TransformAUG)

    SRoUDA_Mate_step(args,train_source_dataset,train_target_RMA,val_dataset,num_classes)



if __name__ == '__main__':
    main()
    # model_path='dan/office31_A2W_MDD/best.pth'
