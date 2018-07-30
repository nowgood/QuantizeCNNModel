# coding=utf-8
import torch
import logging


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # pred: torch.Size([128, 5])
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # 转置
        # pred: torch.Size([5, 128])

        # batch_size 128 target: torch.Size([128]),
        # 也就是说 target 不是 one-hot 编码, 而是 class id
        target = target.view(1, -1).expand_as(pred)
        # [128] =>view=> [1, 128] =>expand_as[5, 128]=>[5, 128]
        correct = pred.eq(target)  # eq: Computes element-wise equality

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_logger(logger_name="nowgood", filename=None, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    if filename is not None:
        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(ch)

    # disable logger
    # logger.setLevel(logger.disabled)

    return logger
