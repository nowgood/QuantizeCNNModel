# coding=utf-8
from utils.meter import AverageMeter, accuracy
from quantize.quantize_fn import QuantizeWeightOrActivation
import torch
import time


def validate(model, val_loader, criterion, gpu, full_precision=True,
             epoch=0, summary_writer=None, name_prefix=None, print_freq=20):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    loss_name = "val/loss"
    prec1_name = "val/top-1"
    prec5_name = "val/top-5"
    if name_prefix is not None:
        name_prefix = ''.join((name_prefix, '-'))
        loss_name = ''.join((name_prefix, loss_name))
        prec1_name = ''.join((name_prefix, prec1_name))
        prec5_name = ''.join((name_prefix, prec5_name))

    # 进入 eval 状态
    model.eval()

    if not full_precision:
        qw = QuantizeWeightOrActivation()  # 1, 创建量化器
        model.apply(qw.quantize)  # 2, 量化权重, 保存全精度权重和量化梯度

    with torch.no_grad():
        start = time.time()
        for i, (data, target) in enumerate(val_loader):
            if gpu is not None:
                data = data.cuda(gpu, non_blocking=True)

            # batch_size 128时, target size 为 torch.Size([128])
            target = target.cuda(gpu, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1[0], data.size(0))
            top5.update(prec5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

        if summary_writer is not None:
            summary_writer.add_scalar(loss_name, losses.avg, epoch)
            summary_writer.add_scalar(prec1_name, top1.avg, epoch)
            summary_writer.add_scalar(prec5_name, top5.avg, epoch)

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    if not full_precision:
        model.apply(qw.restore)  # 第3步, 恢复全精度权重

    return top1.avg

