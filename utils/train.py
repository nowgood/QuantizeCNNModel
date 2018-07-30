# coding=utf-8
import time
import torch
import os
import shutil
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from collections import defaultdict

from utils.val import validate
from utils.meter import AverageMeter, accuracy
from quantize.quantize_fn import QuantizeWeightOrActivation

best_prec1 = 0


def train(args, model, criterion, optimizer, lr_scheduler, train_loader, train_sampler, val_loader):
    global best_prec1

    # 加载日志 summary_writer
    summary_writer = SummaryWriter(args.checkpoint)
    is_full_precision = True if args.mode == 0 else False

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # 调整学习率
        lr_scheduler.step()

        # 训练一个 epoch
        _train(model, train_loader, criterion, optimizer, args.gpu, is_full_precision, epoch, summary_writer)
        # 训练一个 epoch 后, 在验证集上评估
        prec1 = validate(model, val_loader, criterion, args.gpu, is_full_precision, epoch, summary_writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.checkpoint)

    summary_writer.close()


def _train(model, train_loader, criterion, optimizer, gpu=None, full_precision=True,
           epoch=0, summary_writer=None, log_per_epoch=100, print_freq=30):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    if not full_precision:
        qw = QuantizeWeightOrActivation()   # 第一步, 创建量化器

    end = time.time()

    # 用于控制 tensorboard 的显示频率
    interval = len(train_loader) / log_per_epoch
    summary_point = [interval * split for split in torch.arange(0, log_per_epoch)]

    for i, (data, target) in enumerate(train_loader):
        data_time.update(time.time() - end)  # measure checkpoint.pth data loading time

        if gpu is not None:
            data = data.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        if not full_precision:
            model.apply(qw.quantize)  # 第二步, 量化权重, 保存全精度权重和量化梯度

        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1[0], data.size(0))
        top5.update(prec5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        if not full_precision:
            model.apply(qw.restore)  # 第三步, 反向传播后, 模型梯度计算后, 恢复全精度权重
            model.apply(qw.update_grad)  # 第四步, 使用之前存储的量化梯度乘上反向传播的梯度

        # 第五步, 使用更新的梯度更新权重
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # 控制台
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

        if summary_writer and (i in summary_point):
            step = i/interval + (epoch - 1) * log_per_epoch
            summary_writer.add_scalar("loss/train_loss", loss, step)
            summary_writer.add_scalar("train/top-1", top1.avg, step)
            summary_writer.add_scalar("train/top-5", top5.avg, step)


def save_checkpoint(state, is_best, model_dir,  name_prefix=None,
                    checkpoint_name='checkpoint.pth.tar',
                    mode_best_name='model_best.pth.tar',):
    if name_prefix is not None:
        name_prefix = ''.join((name_prefix, '-'))
    else:
        name_prefix = ''

    checkpoint = os.path.join(model_dir, name_prefix, checkpoint_name)
    model_best = os.path.join(model_dir, name_prefix, mode_best_name)

    torch.save(state, checkpoint)
    if is_best:
        shutil.copyfile(checkpoint, model_best)