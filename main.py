import argparse
import os
import shutil
import time
import random

import sys
sys.path.insert(1, '../vision/')
sys.path.append('../utils/')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.losses import FocalLoss

from datetime import datetime
from utils import create_if_not_exists as cine
from utils import Tee
import pickle
from resnet50 import *
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
        help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
        help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
        help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
        metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
        metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=300, type=int,
        metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
        help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
        help='evaluate model on validation set')
parser.add_argument('--train_val', default='val', type=str,
        help='evaluate on train or val')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
        help='use pre-trained model')
parser.add_argument('--nclasses', default='200', type=int,
        help='number of classes')
parser.add_argument('--manualSeed', default='0', type=int,
        help='manual seed')
parser.add_argument('--pool_name', default='GAP', type=str,
        help='pool name: GAP, GMP, KMP, LPP, SMP, MXP, GTP, STP, LAEP, STP')
parser.add_argument('--lr', default='0.001, 30', nargs='+', type=float,
        help='learning rate schedule, lr1, epoch1, lr2, epoch2,...')

best_prec1 = 0
time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

cine('logs')
Tee('logs/cmd_log_{}'.format(time_string), 'w')
features = []

def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    lrs = []
    args.epochs = 0
    for lr, epoch in zip(args.lr[0::2], args.lr[1::2]):
        lrs.extend([lr] * int(epoch))
        args.epochs += int(epoch)
    args.lr = lrs

    # replicable results from fixed seed.
    random.seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    model = resnet50(pretrained=args.pretrained, num_classes=args.nclasses, pool_name=args.pool_name)

    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)

            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = model.cuda()
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize(512),
                transforms.RandomResizedCrop(448),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ]))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(448),
                transforms.ToTensor(),
                normalize,
                ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        if args.train_val == "train":
            validate(train_loader, model, criterion, args.epochs-1)
        elif args.train_val == "val":
            validate(val_loader, model, criterion, args.epochs-1)
        else:
            print("wrong train_val flag")
            return

        pickle.dump(features, open('checkpoints/features_{}.pkl'.format(time_string), 'wb'))
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
            })



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target, paths) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        inputs_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, _= model(inputs_var.cuda(), epoch)
        loss = criterion(output.cuda(), target_var.cuda())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5))


    print(' * Train Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f}'
            .format(top1=top1, top5=top5, loss=losses))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, target, paths) in enumerate(val_loader):
            target = target.cuda()
            inputs_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(target)

            # compute output
            output, ft_out= model(inputs_var.cuda(), epoch)
            loss = criterion(output.cuda(), target_var.cuda())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i > 0 and i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))

        if epoch == args.epochs-1 or args.evaluate:
            batch_size = target.size(0)
            for idx in range(batch_size):
                feat = {}
                feat['name'] = paths[idx]
                feat['gt'] = target[idx]
                feat['ft'] = output[idx].cpu().numpy()
                feat['penu'] = ft_out[0][idx].cpu().numpy()
                feat['pool'] = ft_out[1][idx].cpu().numpy()
                feat['image'] = inverse_normalization(inputs[idx].numpy())
                features.append(feat)

    print(' * Val Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.3f}'
            .format(top1=top1, top5=top5, loss=losses))

    return top1.avg


def save_checkpoint(state, filename='*_checkpoint.pth.tar'):
    work_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    save_dir = os.path.join('/mv_users/peiguo/checkpoints/', work_dir)
    cine(save_dir)
    symlink = './checkpoints'
    if not os.path.exists(symlink):
        os.symlink(save_dir, symlink)

    filename = filename.replace('*', time_string)
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)


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


def adjust_learning_rate(optimizer, epoch):
    optimizer.param_groups[0]['lr'] = args.lr[epoch]
    for param_group in optimizer.param_groups:
        print(len(param_group["params"]), param_group['lr'])

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def inverse_normalization(img, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]):

    for i, m, s in zip(img, mean, std):
        i *= s
        i += m
    return img


if __name__ == '__main__':
    main()
