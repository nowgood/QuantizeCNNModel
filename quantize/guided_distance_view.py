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
from tensorboardX import SummaryWriter
from collections import defaultdict
import time

from utils.train_val import save_checkpoint, validate
from utils.data_loader import load_train_data, load_val_data
from utils.meter import AverageMeter, accuracy
from quantize.quantize_method import quantize_activations_gemm
from net import net_quantize_activation


def guided(args):
    best_low_prec1 = 0
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

    def guided_train(summary_writer, log_per_epoch=100, print_freq=20):

        batch_time = AverageMeter()
        data_time = AverageMeter()

        low_prec_losses = AverageMeter()
        low_prec_top1 = AverageMeter()
        low_prec_top5 = AverageMeter()
        distance_meter = AverageMeter()

        # 状态转化为训练
        low_prec_model.train()
        full_prec_model.eval()

        end = time.time()

        # 用于控制 tensorboard 的显示频率
        interval = len(train_loader) // log_per_epoch
        summary_point = [interval * split for split in torch.arange(log_per_epoch)]

        for i, (input, target) in enumerate(train_loader):
            # measure checkpoint.pth data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)

            # target 必须要转为 cuda 类型
            # If ``True`` and the source is in pinned memory(固定内存),
            # the copy will be asynchronous(异步) with respect to the host
            target = target.cuda(args.gpu, non_blocking=True)

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
                """
                RuntimeError: arguments are located on different GPUs
                解决方法在于手动将feature map都搬到同一个 GPU 上
                """
                layer3 = (quantize_activations_gemm(low_prec_feature_map1[cudaid]) -
                          quantize_activations_gemm(full_prec_feature_map1[cudaid])).norm(p=args.norm) / num_layer3_features
                layer4 = (quantize_activations_gemm(low_prec_feature_map2[cudaid]) -
                          quantize_activations_gemm(full_prec_feature_map2[cudaid])).norm(p=args.norm) / num_layer4_features
                distance += (layer3 + layer4) / len(low_prec_feature_map1)

            distance *= args.balance

            """Guided Key Point end"""

            low_prec_loss = criterion(low_pre_output, target)
            low_prec_prec1, low_prec_prec5 = accuracy(low_pre_output, target, topk=(1, 5))

            low_prec_losses.update(low_prec_loss.item(), input.size(0))
            low_prec_top1.update(low_prec_prec1[0], input.size(0))
            low_prec_top5.update(low_prec_prec5[0], input.size(0))
            distance_meter.update(distance[0], 1)

            # compute gradient and do SGD step
            low_prec_optimizer.zero_grad()
            low_prec_loss.backward()
            low_prec_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {low_prec_loss.val:.4f} ({low_prec_loss.avg:.4f})\t'
                      'Prec@1 {low_prec_top1.val:.3f} ({low_prec_top1.avg:.3f})\t'
                      'Prec@5 {low_prec_top5.val:.3f} ({low_prec_top5.avg:.3f}) \t'
                      'distance {distance.val:.3f} ({distance.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, low_prec_loss=low_prec_losses, low_prec_top1=low_prec_top1,
                        low_prec_top5=low_prec_top5, distance=distance_meter))

            if summary_writer is not None and (i in summary_point):
                step = i / interval + (epoch - 1) * log_per_epoch
                summary_writer.add_scalar("distance", distance_meter.avg, step)
                summary_writer.add_scalar("loss/low_prec_loss", low_prec_loss, step)
                summary_writer.add_scalar("train_low_prec/top-1", low_prec_top1.avg, step)
                summary_writer.add_scalar("train_low_prec/top-5", low_prec_top5.avg, step)

    # 代码用于使用预训练的ResNet18来同时量化网络权重和激活
    print("=> using imageNet pre-trained model '{}'".format(args.arch))
    # 获取预训练模型参数
    full_prec_model = models.__dict__[args.arch](pretrained=True)
    low_prec_model = net_quantize_activation.__dict__[args.arch]()

    model_dict = low_prec_model.state_dict()
    imagenet_dict = full_prec_model.state_dict()
    model_dict.update(imagenet_dict)
    low_prec_model.load_state_dict(model_dict)

    low_prec_layer4 = low_prec_model._modules.get("layer4")
    full_prec_layer4 = full_prec_model._modules.get("layer4")

    hook_low_prec = low_prec_layer4.register_forward_hook(low_prec_hook)
    hook_full_prec = full_prec_layer4.register_forward_hook(full_prec_hook)

    low_prec_model = gpu_config(low_prec_model)
    full_prec_model = gpu_config(full_prec_model)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    low_prec_optimizer = torch.optim.SGD(low_prec_model.parameters(),
                                         args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

    low_prec_scheduler = torch.optim.lr_scheduler.StepLR(low_prec_optimizer, step_size=args.lr_step, gamma=0.1)

    cudnn.benchmark = True

    val_loader = load_val_data(args.data, args.batch_size, args.workers)
    train_loader, train_sampler = load_train_data(args.data, args.batch_size, args.workers, args.distributed)

    # 加载日志 writer
    writer = SummaryWriter(args.save_dir)

    for epoch in range(args.start_epoch, args.epochs+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        low_prec_scheduler.step()

        # train for one epoch
        guided_train(writer)

        # evaluate on validation set
        low_prec1 = validate(low_prec_model, val_loader, criterion, args.gpu,
                             epoch, writer, name_prefix='low_prec')

        # remember best prec@1 and save low_prec_checkpoint
        is_best_low = low_prec1 > best_low_prec1

        best_low_prec1 = max(low_prec1, best_low_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': low_prec_model.state_dict(),
            'best_prec1': best_low_prec1,
            'optimizer': low_prec_optimizer.state_dict(),
        }, is_best_low, args.save_dir, name_prefix="low_prec")

    # 关闭日志 writer
    writer.close()

    # 去掉钩子

    hook_full_prec.remove()
    hook_low_prec.remove()

