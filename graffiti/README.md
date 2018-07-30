
### Usage: [argparse](http://wiki.jikexueyuan.com/project/explore-python/Standard-Modules/argparse.html) 

```
 每个参数解释如下:

    name or flags - 选项字符串的名字或者列表，例如 foo 或者 -f, --foo。
    action - 命令行遇到参数时的动作，默认值是 store。
    store_const，表示赋值为const；
    append，将遇到的值存储成列表，也就是如果参数重复则会保存多个值;
    append_const，将参数规范中定义的一个值保存到一个列表；
    count，存储遇到的次数；此外，也可以继承 argparse.Action 自定义参数解析；
    nargs - 应该读取的命令行参数个数，可以是具体的数字，或者是?号，当不指定值时对于 Positional argument 使用 default，
            对于 Optional argument 使用 const；
            或者是 * 号，表示 0 或多个参数；
            或者是 + 号表示 1 或多个参数。
    const - action 和 nargs 所需要的常量值。
    default - 不指定参数时的默认值。
    type - 命令行参数应该被转换成的类型。
    choices - 参数可允许的值的一个容器。
    required - 可选参数是否可以省略 (仅针对可选参数)。
    help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
    metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.
    dest - 解析后的参数名称，默认情况下，对于可选参数选取最长的名称，中划线转换为下划线.
```

### Usage imagenet.py 

```
usage: guided.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | resnet | resnet101 |
                        resnet152 | resnet18 | resnet34 | resnet50 | vgg |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --README.md-rate LR
                        initial README.md rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model

```

### use pretrained model to initialize your modified model

```
model_dict = your_model.state_dict()

pretrained_model = models.__dict__[args.arch](pretrained=True)
pretrained_dict = pretrained_model.state_dict()

# 将 pretrained_dict 里不属于 model_dict 的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict)
your_model.load_state_dict(model_dict)
```

### how to get nn.DataParallel model filter weight

```python
low_prec_state_dict = low_prec_model.state_dict()
full_prec_state_dict = full_prec_model.state_dict()
low_prec_norm = low_prec_state_dict['module.layer4.1.conv1.weight'].norm(p=2) + low_prec_state_dict['module.layer4.1.conv2.weight'].norm(p=2)
full_prec_norm = full_prec_state_dict['module.layer4.1.conv1.weight'].norm(p=2) + full_prec_state_dict['module.layer4.1.conv2.weight'].norm(p=2)

l2 = (low_prec_norm + full_prec_norm) * args.balance
```

### torch.topk

```
>>> x = torch.arange(1, 6)
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.])
>>> torch.topk(x, 3)
(tensor([ 5.,  4.,  3.]), tensor([ 4,  3,  2]))
```