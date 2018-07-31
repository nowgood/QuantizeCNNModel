## Quantize CNN Model using PyTorch(python3.5)
 
Implement [Towards Effective Low-bitwidth Convolutional Neural Networks](https://arxiv.org/abs/1711.00205)

```
@InProceedings{Zhuang_2018_CVPR,
author = {Zhuang, Bohan and Shen, Chunhua and Tan, Mingkui and Liu, Lingqiao and Reid, Ian},
title = {Towards Effective Low-Bitwidth Convolutional Neural Networks},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

### 下载和配置

```bash
git clone https://github.com/nowgood/QuantizeCNNModel.git && cd QuantizeCNNModel
pip install -r requirements.txt
echo export PYTHONPATH=$PYTHONPATH:`pwd` >> ~/.bashrc
source  ~/.bashrc
```

### 使用方法

使用如下命令查看函数使用方法 

```
python guided.py -h 
```



然后使用 tensorboard 查看训练过程

```
# QuantizeCNNModel 目录下
tensorboard --logdir model/xxx/ 
```
然后就可以在 `http:localhost:6006` 查看训练的损失值和精确度， 以及每个epoch的在验证集上的精确度

![top5](https://github.com/nowgood/QuantizeCNNModel/raw/master/data/WandA_lr0.01_scalar2.5.png)

### 训练方法

训练模式选择:

       0: full precision training from scratch
       1: only quantize weight
       2. quantize activation using quantized weight to init model
       3. joint quantize weight and activation from pre-trained imageNet model
       4. guided quantize weight and activation from pre-trained imageNet model


**单卡训练**

```
python guided.py \
    --arch resnet18 \
    --mode 3 \
    --workers 16 \
    --epochs 35 \
    --checkpoint model/WandA_lr0.001_scalar2.5 \
    --lr 0.001 \
    --data /media/wangbin/8057840b-9a1e-48c9-aa84-d353a6ba1090/ImageNet_ILSVRC2012/ILSVRC2012 \
    > log/WandA_lr_0.001_scalar2.5_20180719.log 2>&1 &
```

### 量化权重

单机多卡训练, 如： 使用 8 个GPU的后 4 个GPU来训练25个epoch

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --arch resnet18 \
    --mode 1 \
    --workers 16 \
    --epochs 25 \
    --batch_size 1024\
    --device_ids 0 1 2 3 \
    --lr 0.0001 \
    --checkpoint model/W_lr0.0001_epoch25 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    |tee  model/W_lr_1e-4_epoch25.log 2>&1
``` 

### 使用量化权重的参数来初始化量化激活的网络

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --arch resnet18 \
    --mode 2 \
    --workers 16 \
    --epochs 35 \
    --batch_size 1024\
    --device_ids 0 1 2 3 \
    --lr 0.01 \
    --weight_quantized model/W_lr1e-4_epoch2/model_best.pth.tar \
    --checkpoint model/AafterW_lr1e-2_epoch35 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    |tee  model/AafterW_lr1e-2_epoch35.log 2>&1
```

**resume**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --arch resnet18 \
    --mode 2 \
    --workers 16 \
    --epochs 35 \
    --batch_size 1024\
    --device_ids 0 1 2 3 \
    --lr 0.001 \
    --resume model/AafterW_lr1e-3_epoch35/checkpoint.pth.tar \
    --weight_quantized model/W_lr1e-4_epoch2/model_best.pth.tar \
    --checkpoint model/AafterW_lr1e-3_epoch35 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    |tee  model/AafterW_lr1e-3_epoch35.log 2>&1
```

### 同时量化权重和激活

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python guided.py \
    --mode 3
    --arch resnet18 \
    --workers 16 \
    --epochs  35 \
    --batch-size 800 \
    --pretrained \
    --device_ids 0 1 2 3 \
    --lr 0.01 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    --checkpoint model/AandW_lr0.01_epoch35 \
    | tee AandW_lr0.01_epoch35.log 2>&1 
```

### 使用 guidance 信号来同时量化权重和激活

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --arch resnet18
    --mode 4 \
    --workers 16 \
    --epochs  50 \
    --batch-size 512 \
    --device_ids 0 1 2 3 \
    --balance 2 \
    --lowlr 0.001 \
    --fulllr 0.001 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    --checkpoint /home/user/wangbin/quantizednn/model/WandA_guided_balance2_lr1e-3_lr1e-3_epoch50 \
    | tee model/log.WandA_guided_balance2_lr1e-3_lr1e-3_epoch50 2>&1 
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --arch resnet18
    --mode 4 \
    --workers 16 \
    --epochs  50 \
    --batch-size 512 \
    --device_ids 0 1 2 3 \
    --balance 2 \
    --lowlr 0.001 \
    --fulllr 0.001 \
    --norm 2 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    --checkpoint /home/user/wangbin/quantizednn/model/WandA_guided_balance2_lr1e-3_lr1e-3_epoch50 \
    | tee model/log.WandA_guided_balance2_lr1e-3_lr1e-3_epoch50 2>&1 
```

```bash
--weight-quantized
/home/wangbin/Desktop/uisee/model_quantize/w_lr1e-4_epoch2_QCONV/checkpoint.pth.tar
```