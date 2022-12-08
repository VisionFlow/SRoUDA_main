#!/usr/bin/env bash
# ResNet50, Office31, Single Source
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/office31 -d Office31 -s A -t W -a resnet50 --epochs 20 --mask_ratio 0.25 --patch-size 8  --rma 0.1 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_A2W
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/office31 -d Office31 -s D -t W -a resnet50 --epochs 20 --mask_ratio 0.25 --patch-size 8 --rma 0.1 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_D2W
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/office31 -d Office31 -s W -t D -a resnet50 --epochs 20 --mask_ratio 0.25 --patch-size 8 --rma 0.1 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_W2D
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/office31 -d Office31 -s A -t D -a resnet50 --epochs 20 --mask_ratio 0.25 --patch-size 8 --rma 0.1 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_A2D
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/office31 -d Office31 -s D -t A -a resnet50 --epochs 20 --mask_ratio 0.25 --patch-size 8 --rma 0.1 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_D2A
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/office31 -d Office31 -s W -t A -a resnet50 --epochs 20 --mask_ratio 0.25 --patch-size 8 --rma 0.1 --bottleneck-dim 1024 --seed 1 --log logs/mdd/Office31_W2A


# Digits
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/digits -d Digits -s MNIST -t USPS --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 28 --no-hflip --norm-mean 0.5 --norm-std 0.5 -a lenet --no-pool --lr 0.01 -b 128 -i 2500 --mask_ratio 0.85 --patch-size 2 --rma 0.1 --scratch --seed 0 --log logs/mdd/MNIST2USPS
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/digits -d Digits -s USPS -t MNIST --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 28 --no-hflip --norm-mean 0.5 --norm-std 0.5 -a lenet --no-pool --lr 0.01 -b 128 -i 2500 --mask_ratio 0.85 --patch-size 2 --rma 0.1 --scratch --seed 0 --log logs/mdd/USPS2MNIST
CUDA_VISIBLE_DEVICES=0 python mdd.py ../data/digits -d Digits -s SVHNRGB -t MNISTRGB --train-resizing 'res.' --val-resizing 'res.' \
  --resize-size 32 --no-hflip --norm-mean 0.5 0.5 0.5 --norm-std 0.5 0.5 0.5 -a dtn --no-pool --lr 0.01 -b 128 -i 2500 --mask_ratio 0.85 --patch-size 2 --rma 0.1 --scratch --seed 0 --log logs/mdd/SVHN2MNIST
