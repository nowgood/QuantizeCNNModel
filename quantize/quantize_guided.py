# coding=utf-8
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from net import quantize_net
from tensorboardX import SummaryWriter
from collections import defaultdict
import time

from utils.train import save_checkpoint
from utils.data_loader import load_train_data, load_val_data
from utils.val import validate
from utils.meter import AverageMeter, accuracy
from quantize.quantize_fn import QuantizeWeightOrActivation


def guided(args):
    best_low_prec1 = 0
    best_full_prec1 = 0
    full_prec_feature_map1 = defaultdict(torch.Tensor)
    full_prec_feature_map2 = defaultdict(torch.Tensor)
    low_prec_feature_map1 = defaultdict(torch.Tensor)
    low_prec_feature_map2 = defaultdict(torch.Tensor)

    def full_prec_hook(module, input, output):
        # 一定要写成 input[0].data.clone()
        # 而不能写成 input[0].clone(), 否则报错
        # RuntimeError: Trying to backward through the graph a second time,
        # but the buffers have already been freed. Specify retain_graph=True
        # when calling backward the first time
        cudaid = int(repr(output.device)[-2])
        full_prec_feature_map1[cudaid] = input[0].data.clone()
        full_prec_feature_map2[cudaid] = output.data.clone()

    def low_prec_hook(module, input, output):
        cudaid = int(repr(output.device)[-2])
        low_prec_feature_map1[cudaid] = input[0].data.clone()
        low_prec_feature_map2[cudaid] = output.data.clone()

    def gpu_config(model):
        if args.gpu is not None:  # 指定GPU
            model = model.cuda(args.gpu)
        elif args.distributed:  # 集群训练（多机器）
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

        else:  # 单机训练（单卡或者多卡）
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                # 一机多卡时, 多 GPU 训练, 指定要用到 GPU 的 ids
                model = torch.nn.DataParallel(model, args.device_ids).cuda()
        return model

    def guided_train(args, train_loader, low_prec_model, full_prec_model, criterion,
                     low_prec_optimizer, full_prec_optimizer, epoch, summary_writer, log_per_epoch=100, print_freq=20):

        batch_time = AverageMeter()
        data_time = AverageMeter()

        low_prec_losses = AverageMeter()
        low_prec_top1 = AverageMeter()
        low_prec_top5 = AverageMeter()

        full_prec_losses = AverageMeter()
        full_prec_top1 = AverageMeter()
        full_prec_top5 = AverageMeter()

        # 状态转化为训练
        low_prec_model.train()
        full_prec_model.train()

        # 第一步, 创建量化器
        qw = QuantizeWeightOrActivation()

        end = time.time()

        # 用于控制 tensorboard 的显示频率
        interval = len(train_loader) / log_per_epoch
        summary_point = [interval * split for split in torch.arange(log_per_epoch)]

        for i, (input, target) in enumerate(train_loader):
            # measure checkpoint.pth data loading time
            data_time.update(time.time() - end)

            # 这一步为何 ?
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            # target 必须要转为 cuda 类型
            # If ``True`` and the source is in pinned memory(固定内存),
            # the copy will be asynchronous(异步) with respect to the host
            target = target.cuda(args.gpu, non_blocking=True)

            # 第二步, 量化权重, 保存全精度权重和量化梯度
            low_prec_model.apply(qw.quantize)

            full_prec_feature_map1.clear()
            low_prec_feature_map1.clear()
            full_prec_feature_map2.clear()
            low_prec_feature_map2.clear()

            # compute low_pre_output
            low_pre_output = low_prec_model(input)
            full_pre_output = full_prec_model(input)

            """Guided Key Point start"""

            # 将 distance 和 feature map放在同一个一gpu上
            distance = torch.tensor([0.0]).cuda(args.gpu, non_blocking=True)
            num_layer3_features = 1
            for dim in full_prec_feature_map1[0].size():
                num_layer3_features *= dim

            num_layer4_features = 1
            for dim in full_prec_feature_map2[0].size():
                num_layer4_features *= dim

            for cudaid in full_prec_feature_map1:
                # 手动将feature map都搬到同一个 GPU 上
                full_prec_feature_map1[cudaid] = full_prec_feature_map1[cudaid].cuda(args.gpu, non_blocking=True)
                low_prec_feature_map1[cudaid] = low_prec_feature_map1[cudaid].cuda(args.gpu, non_blocking=True)
                full_prec_feature_map2[cudaid] = full_prec_feature_map2[cudaid].cuda(args.gpu, non_blocking=True)
                low_prec_feature_map2[cudaid] = low_prec_feature_map2[cudaid].cuda(args.gpu, non_blocking=True)

            for cudaid in low_prec_feature_map1:
                """RuntimeError: arguments are located on different GPUs
                   解决方法在于手动将feature map都搬到同一个 GPU 上
                """
                layer3 = (qw.quantize_activations(low_prec_feature_map1[cudaid]) -
                          full_prec_feature_map1[cudaid]).norm(p=1) / num_layer3_features
                layer4 = (qw.quantize_activations(low_prec_feature_map2[cudaid]) -
                          full_prec_feature_map2[cudaid]).norm(p=1) / num_layer4_features
                distance += (layer3 + layer4)

            distance *= args.balance
            low_prec_loss = criterion(low_pre_output, target) + distance
            full_prec_loss = criterion(full_pre_output, target) + distance

            low_prec_prec1, low_prec_prec5 = accuracy(low_pre_output, target, topk=(1, 5))
            full_prec_prec1, full_prec_prec5 = accuracy(full_pre_output, target, topk=(1, 5))

            low_prec_losses.update(low_prec_loss.item(), input.size(0))
            low_prec_top1.update(low_prec_prec1[0], input.size(0))
            low_prec_top5.update(low_prec_prec5[0], input.size(0))

            full_prec_losses.update(full_prec_loss.item(), input.size(0))
            full_prec_top1.update(full_prec_prec1[0], input.size(0))
            full_prec_top5.update(full_prec_prec5[0], input.size(0))

            # compute gradient and do SGD step
            low_prec_optimizer.zero_grad()
            full_prec_optimizer.zero_grad()

            low_prec_loss.backward()
            full_prec_loss.backward()

            # 第三步, 反向传播后, 模型梯度计算后, 恢复全精度权重
            low_prec_model.apply(qw.restore)
            # 第四步, 使用之前存储的量化梯度乘上反向传播的梯度
            low_prec_model.apply(qw.update_grad)

            # 第五步, 使用更新的梯度更新权重
            low_prec_optimizer.step()
            full_prec_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('=>balance {} \t distance: {}'.format(args.balance, distance))
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {low_prec_loss.val:.4f} ({low_prec_loss.avg:.4f})\t'
                      'Prec@1 {low_prec_top1.val:.3f} ({low_prec_top1.avg:.3f})\t'
                      'Prec@5 {low_prec_top5.val:.3f} ({low_prec_top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, low_prec_loss=low_prec_losses, low_prec_top1=low_prec_top1,
                        low_prec_top5=low_prec_top5))

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {full_prec_loss.val:.4f} ({full_prec_loss.avg:.4f})\t'
                      'Prec@1 {full_prec_top1.val:.3f} ({full_prec_top1.avg:.3f})\t'
                      'Prec@5 {full_prec_top5.val:.3f} ({full_prec_top5.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, full_prec_loss=full_prec_losses, full_prec_top1=full_prec_top1,
                        full_prec_top5=full_prec_top5))

            if summary_writer is not None and (i in summary_point):
                step = i / interval + (epoch - 1) * log_per_epoch
                summary_writer.add_scalar("distance", distance, step)
                summary_writer.add_scalar("loss/low_prec_loss", low_prec_loss, step)
                summary_writer.add_scalar("train_low_prec/top-1", low_prec_top1.avg, step)
                summary_writer.add_scalar("train_low_prec/top-5", low_prec_top5.avg, step)

                summary_writer.add_scalar("loss/full_prec_loss", full_prec_loss, step)
                summary_writer.add_scalar("train_full_prec/top-1", full_prec_top1.avg, step)
                summary_writer.add_scalar("train_full_prec/top-5", full_prec_top5.avg, step)

    # 1. 创建全精度模型和低精度模型
    low_prec_model = quantize_net.resnet18()
    full_prec_model = models.__dict__[args.arch](pretrained=True)

    if args.weight_quantized:
        print("=> using quantize-weight model '{}'".format(args.arch))
        if os.path.isfile(args.weight_quantized):
            print("=> loading weight_quantized model '{}'".format(args.weight_quantized))
            model_dict = low_prec_model.state_dict()
            quantized_model = torch.load(args.weight_quantized)
            pretrained__dict = {k[7:]: v for k, v in quantized_model['state_dict'].items()
                                if k in low_prec_model.state_dict()}
            model_dict.update(pretrained__dict)
            low_prec_model.load_state_dict(model_dict)
            print("=> loaded weight_quantized '{}'".format(args.weight_quantized))
        else:
            print("=> no  quantize-weight model found at '{}'".format(args.weight_quantized))
    else:
        # 代码用于使用预训练的ResNet18来同时量化网络权重和激活
        print("=> using imageNet pre-trained model '{}'".format(args.arch))
        # 获取预训练模型参数
        model_dict = low_prec_model.state_dict()
        imagenet_model = models.__dict__[args.arch](pretrained=True)
        imagenet_dict = {k: v for k, v in imagenet_model.state_dict().items()
                         if k in model_dict}
        model_dict.update(imagenet_dict)
        low_prec_model.load_state_dict(model_dict)

    if not args.evaluate:
        low_prec_layer4 = low_prec_model._modules.get("layer4")
        full_prec_layer4 = full_prec_model._modules.get("layer4")

        hook_low_prec = low_prec_layer4.register_forward_hook(low_prec_hook)
        hook_full_prec = full_prec_layer4.register_forward_hook(full_prec_hook)

    low_prec_model = gpu_config(low_prec_model)
    full_prec_model = gpu_config(full_prec_model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    low_prec_optimizer = torch.optim.SGD(low_prec_model.parameters(),
                                         args.lowlr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
    full_prec_optimizer = torch.optim.SGD(low_prec_model.parameters(),
                                          args.fulllr,
                                          momentum=args.momentum,
                                          weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        full_prec_resume = os.path.join(args.resume, "full_prec_checkpoint.pth.tar")
        low_prec_resume = os.path.join(args.resume, "low_prec_checkpoint.pth.tar")
        if os.path.isfile(full_prec_resume) and os.path.isfile(low_prec_resume):
            print("=> loading low_prec_checkpoint from '{}' and '{}'".format(full_prec_resume,
                                                                             low_prec_model))
            full_prec_checkpoint = torch.load(args.resume)
            low_prec_checkpoint = torch.load(args.resume)

            args.start_epoch = low_prec_checkpoint['epoch']
            # 模型的最好精度
            best_low_prec1 = low_prec_checkpoint['best_low_prec1']
            best_full_prec1 = full_prec_checkpoint['best_full_prec1']

            low_prec_model.load_state_dict(low_prec_checkpoint['state_dict'])
            full_prec_model.load_state_dict(full_prec_checkpoint['state_dict'])

            low_prec_optimizer.load_state_dict(low_prec_checkpoint['optimizer'])
            full_prec_optimizer.load_state_dict(full_prec_checkpoint['optimizer'])

            print("=> loaded low_prec_checkpoint from '{}' and '{}' (epoch {})".format(
                full_prec_resume, low_prec_model, low_prec_checkpoint['epoch']))
        else:
            print("=> no checkpoint found at directory'{}'".format(args.resume))

    cudnn.benchmark = True

    val_loader = load_val_data(args.data, args.batch_size, args.workers)
    train_loader, train_sampler = load_train_data(args.data, args.batch_size, args.workers, args.distributed)

    # 加载日志 writer
    writer = SummaryWriter(args.checkpoint)

    # 调整学习率
    full_prec_scheduler = torch.optim.lr_scheduler.StepLR(full_prec_optimizer, step_size=10, gamma=0.1)
    low_prec_scheduler = torch.optim.lr_scheduler.StepLR(low_prec_optimizer, step_size=10, gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        full_prec_scheduler.step()
        low_prec_scheduler.step()

        # train for one epoch
        guided_train(args, train_loader, low_prec_model, full_prec_model, criterion,
                     low_prec_optimizer, full_prec_optimizer, epoch, writer)

        # evaluate on validation set
        low_prec1 = validate(val_loader, low_prec_model, criterion, args.gpu, False,
                             epoch, writer, name_prefix='low_prec')
        full_prec1 = validate(val_loader, full_prec_model, criterion, args.gpu, True,
                              epoch, writer, name_prefix='full_prec')

        # remember best prec@1 and save low_prec_checkpoint
        is_best_low = low_prec1 > best_low_prec1
        is_best_full = full_prec1 > best_full_prec1

        best_low_prec1 = max(low_prec1, best_low_prec1)
        best_full_prec1 = max(full_prec1, best_full_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': low_prec_model.state_dict(),
            'best_prec1': best_low_prec1,
            'low_prec_optimizer': low_prec_optimizer.state_dict(),
        }, is_best_low, args.checkpoint, name_prefix="low_prec")

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': full_prec_model.state_dict(),
            'best_prec1': best_full_prec1,
            'full_prec_optimizer': full_prec_optimizer.state_dict(),
        }, is_best_full, args.checkpoint, name_prefix="full_prec")

    # 关闭日志 writer
    writer.close()

    # 去掉钩子
    if not args.evaluate:
        hook_full_prec.remove()
        hook_low_prec.remove()
