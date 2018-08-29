# coding=utf-8

"""
1. 量化权重建议:
    1). 学习率最高设置为 0.001, 0.0001可以很快的收敛, 最很好的选择, 训练个 2~5个 epoch 就好

2. 权重和激活同时量化注意事项:
    1). 学习率设置不能大于 0.01(学习率最大设置 0.01), 当学习率设置为0.01时, 模型可以很好的微调,
    2). 当学习率设置为 0.1 时, 训练几十个batch之后, 准确率为 千分之一 和 千分之五
    3). 学习率设置为 0.01 时,大约 5~8 epoch降低一次学习率(除以10)比较好, 然后训练大约 30~40 epoch就好
    4). 当学习率设置为 0.001 时, 大约 14~16 epoch 降低一次学习率比较好, 然后训练大约 30~40 epoch就好

3. 训练模式(mode):
       0: full precision training from scratch
       1: only quantize_tanh weight
       2. quantize_tanh activation using quantized weight to init model
       3. joint quantize_tanh weight and activation from pre-trained imageNet model
       4. guided quantize_tanh weight and activation from pre-trained imageNet model

"""

import argparse
import torchvision.models as models
import warnings
import random
import os
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.train_val import train, save_checkpoint, validate
from utils.data_loader import load_train_data, load_val_data
from quantize import quantize_guided
from quantize.quantize_method import quantize_weights_bias_gemm
from net import net_quantize_activation, net_quantize_weight
from tensorboardX import SummaryWriter


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', metavar='DIR', help='path to dataset', required=True)
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--workers', default=16, type=int, metavar='N',  # 修改为电脑cpu支持的线程数
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=35, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', action='store_true',
                    help='resume training using save-dir checkpoint (default: False)')
# 如果是验证模型, 设置为True就好, 训练时值为False
parser.add_argument('--evaluate', default='', type=str,
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

parser.add_argument('--device-ids', default=[0], type=int, nargs='+',
                    help='GPU ids to be used e.g 0 1 2 3')
parser.add_argument('--weight-quantized', default='', type=str, help="quantize_tanh weight model path")
parser.add_argument('--save-dir', default='model', type=str, help='directory to save trained model', required=True)
parser.add_argument('--mode', default=3, type=int, help='model quantized mode', required=True)
# l1 norm balance 设置为1或者0.1比较好, l2 norm balance 设置为100(~0.034) ~ 500 比较好
parser.add_argument('--norm', default=1, type=int, help='feature map norm, default 1')
parser.add_argument('--balance', default=100, type=float, help='balancing parameter (default: 100)')
# 论文中初始学习率 0.001, 每 10 epoch 除以 10, 这在只量化权重时候可以
# 在同时量化权重和激活时, 当使用0.001时, 我们可以观测到权重的持续上升
# 或许可以将初始学习率调为 0.01, 甚至 0.1
# guidance 方法中, 全精度模型的的学习率要小一些, 模型已经训练的很好了, 微调而已
# 不过来低精度模型的学习率可以调高一点
parser.add_argument('--lr', default=0.001, type=float,  # 论文中初始学习率 0.001, 每 10 epoch 除以 10
                    help='initial learning rate')
parser.add_argument('--rate', default=1, type=int,
                    help='guide training method, full_lr = low_lr * rate')

parser.add_argument('--lr-step', default=10, type=int, help='learning rate step scheduler')


args = parser.parse_args()
best_prec1 = 0


def main():
    global best_prec1
    print("\n"
          "=> arch         {: <20}\n"
          "=> init_lr      {: <20}\n"
          "=> lr-step      {: <20}\n"
          "=> momentum     {: <20}\n"
          "=> weight-decay {: <20}\n"
          "=> batch-size   {: <20}\n"
          "=> balance      {: <20}\n"
          "=> save-dir     {: <20}\n".format(
           args.arch, args.lr, args.lr_step, args.momentum, args.weight_decay,
           args.batch_size, args.balance, args.save_dir))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably!, You may see unexpected behavior'
                      ' when restarting from checkpoints.')

    # 下面的 warning 可以看出, 如果指定一个 gpu id, 就不会使用多 gpu 训练
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU, This will completely disable data parallelism.')

    # 多机器训练而不是一机多卡(集群训练模式)
    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size)

    # 根据训练模式加载训练模型
    if args.mode == 0:
        print("=> training mode {}: full precision training from scratch\n".format(args.mode))
        model = models.__dict__[args.arch]()

    elif args.mode == 1:
        print("=> training mode {}: quantize weight only\n".format(args.mode))
        print("=> loading imageNet pre-trained model {}".format(args.arch))
        model = net_quantize_weight.__dict__[args.arch]()
        model_dict = model.state_dict()
        init_model = models.__dict__[args.arch](pretrained=True)
        model_dict.update(init_model.state_dict())
        model.load_state_dict(model_dict)
        print("=> loaded imageNet pre-trained model {}".format(args.arch))

    elif args.mode == 2:
        print("=> training mode {}: quantize activation using quantized weight\n".format(args.mode))
        model = net_quantize_activation.__dict__[args.arch]()
        if os.path.isfile(args.weight_quantized):
            print("=> loading weight quantized model '{}'".format(args.weight_quantized))
            model_dict = model.state_dict()
            quantized_model = torch.load(args.weight_quantized)
            init_dict = {}
            for k, v in quantized_model['state_dict'].items():
                if k in model.state_dict():
                    if k.find("conv") != -1 or k.find("fc") != -1:
                        init_dict[k[7:]] = quantize_weights_bias_gemm(v)
                    else:
                        init_dict[k[7:]] = v

            model_dict.update(init_dict)
            model.load_state_dict(model_dict)
            print("=> loaded weight_quantized '{}'".format(args.weight_quantized))
        else:
            warnings.warn("=> no weight quantized model found at '{}'".format(args.weight_quantized))
            return

    elif args.mode == 3:
        print("=> training mode {}: quantize weight and activation simultaneously\n".format(args.mode))
        print("=> loading imageNet pre-trained model '{}'".format(args.arch))
        # 使用预训练的ResNet18来初始化同时量化网络权重和激活
        model = net_quantize_activation.__dict__[args.arch]()
        # 获取预训练模型参数
        model_dict = model.state_dict()
        init_model = models.__dict__[args.arch](pretrained=True)
        init_dict = {k: v for k, v in init_model.state_dict().items() if k in model_dict}
        model_dict.update(init_dict)
        model.load_state_dict(model_dict)

    elif args.mode == 4:
        print("=> Training mode {}: guided quantize weight and activation "
              "from pre-trained imageNet model {}\n ".format(args.mode, args.arch))

        # quantize_guided.guided(args)
        quantize_guided.guided(args)
        return
    else:
        raise Exception("invalid mode, valid mode is 0~4!!")

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
            """
               list(model.state_dict().keys())[0]
               model 在使用 torch.nn.DataParallel 之前每层的名字, 如 conv1.weight
               model 在使用 torch.nn.DataParallel 之后每层的名字, 如 module.conv1.weight
               如果训练使用并行化, 而验证使用指定GPU的话就会出现问题, 所以需要在指定GPU代码中,添加解决冲突的代码
            """
            model = torch.nn.DataParallel(model, args.device_ids).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step)

    # optionally resume from a checkpoint
    if args.resume:
        print("\n=> resume training from checkpoint")
        checkpoint_filename = os.path.join(args.save_dir, "checkpoint.pth.tar")

        if os.path.isfile(checkpoint_filename):
            print("=> loading checkpoint '{}'".format(checkpoint_filename))
            checkpoint = torch.load(checkpoint_filename)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_filename))

    cudnn.benchmark = True

    val_loader = load_val_data(args.data, args.batch_size, args.workers)

    if args.evaluate:
        if os.path.isfile(args.evaluate):
            print("Loading evaluate model '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            if "state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint['state_dict'])
                print("epoch: {} ".format(checkpoint['epoch']))
            else:
                checkpoint = {''.join(("module.", k)): v for k, v in checkpoint.items() if not k.startswith("module")}
                model.load_state_dict(checkpoint)
            print("Loaded evaluate model '{}'".format(args.evaluate))
        else:
            print("No evaluate mode found at '{}'".format(args.evaluate))
            return
        validate(model, val_loader, criterion, args.gpu)
        return

    train_loader, train_sampler = load_train_data(args.data, args.batch_size, args.workers, args.distributed)

    summary_writer = SummaryWriter(args.save_dir)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        lr_scheduler.step()

        # train for one epoch
        train(model, train_loader, criterion, optimizer, args.gpu, epoch, summary_writer)

        # evaluate on validation set
        prec1 = validate(model, val_loader, criterion, args.gpu, epoch, summary_writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch+1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_dir)

    summary_writer.close()


if __name__ == '__main__':
    main()