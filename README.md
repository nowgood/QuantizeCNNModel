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
python main.py -h 
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


### 量化权重

单机多卡训练, 如： 使用 8 个GPU的后 4 个GPU来训练25个epoch

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --mode 1 \
    --workers 16 \
    --epochs 5 \
    --batch-size 1024\
    --device-ids 0 1 2 3 \
    --lr 0.0001 \
    --lr-step 2 \
    --save-dir model/W_lr1e-4_epoch5 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    |tee  model/W_lr_1e-4_epoch5.log 2>&1
``` 

```

CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --mode 1 \
    --workers 16 \
    --epochs 10 \
    --batch-size 1024\
    --device-ids 0 1 2 3 \
    --lr 0.0001 \
    --lr-step 4 \
    --save-dir model/W_lr1e-4_epoch10 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    |tee  model/W_lr_1e-4_epoch10.log
```   

### 使用量化权重的参数来初始化量化激活的网络

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --mode 2 \
    --workers 16 \
    --epochs 35 \
    --batch-size 1024\
    --device-ids 0 1 2 3 \
    --lr 0.001 \
    --weight-quantized model/W_lr1e-4_epoch2/model_best.pth.tar \
    --save-dir model/AafterW_lr1e-2_epoch35 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    |tee  model/AafterW_lr1e-2_epoch35.log
```

**resume**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --mode 2 \
    --workers 16 \
    --epochs 35 \
    --batch-size 1024\
    --device-ids 0 1 2 3 \
    --lr 0.001 \
    --resume \
    --weight-quantized model/W_lr1e-4_epoch2/model_best.pth.tar \
    --save-dir model/AafterW_lr1e-3_epoch35 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    | tee  model/AafterW_lr1e-3_epoch35.log
```

### 同时量化权重和激活

```
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py \
    --mode 3 \
    --arch resnet18 \
    --workers 16 \
    --epochs  35 \
    --batch-size 512 \
    --device-ids 0 1 2 3 \
    --lr 0.001 \
    --lr-step 10 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    --save-dir model/AandW_lr1e-3_epoch35 \
    | tee AandW_1e-3_epoch35.log
```

### 使用 guidance 信号来同时量化权重和激活

```bash
CUDA_VISIBLE_DEVICES=3,4,5,6 python main.py \
    --mode 4 \
    --workers 16 \
    --epochs  35 \
    --batch-size 512 \
    --device-ids 0 1 2 3\
    --balance 0.1 \
    --lr 0.001 \
    --rate 1 \
    --norm 1 \
    --data /home/user/wangbin/datasets/ILSVRC2012  \
    --save-dir /home/user/wangbin/quantizednn/model/guided_balance0.1_lr1e-3_rate1_epoch35 \
    | tee model/guided_balance0.11_lr1e-3_rate1_epoch35.log
```

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python main.py  \
   --mode 4  \
   --workers 16  \
   --epochs  35  \
   --batch-size 384  \
   --device-ids 0 1 2  \
   --balance 0.1  \
   --lr 0.001  \
   --rate 1   \
   --norm 1  \
   --data /home/user/wangbin/datasets/ILSVRC2012 \
   --resume  \
   --save-dir /home/user/wangbin/quantizednn/model/guided_balance0.1_lr1e-3_rate1_epoch35 \
   | tee model/guided_balance0.1_lr1e-3_rate1_epoch35_resume.log
```

#### view distance

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py  \
   --mode 4  \
   --workers 16  \
   --epochs  35  \
   --batch-size 512  \
   --device-ids 0 1 2 3  \
   --balance 0.1  \
   --lr 0.001  \
   --rate 1   \
   --norm 1  \
   --data /home/user/wangbin/datasets/ILSVRC2012 \
   --save-dir /home/user/wangbin/quantizednn/model/guided_balance0.1_lr1e-3_rate1_epoch35_view 
```