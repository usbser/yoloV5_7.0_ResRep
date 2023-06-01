# 修改记录
## yolov5s.yaml
增加参数 
```iscompactor: True```

## yolo.py  
主要修改parse_model(d, ch)
### 将是否使用resrep的布尔变量通过d传入
```iscompactor = d.get('iscompactor')```
### 定义cur_conv_idx用于为所有的卷积层进行标记
```cur_conv_idx = 0```
### 在for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):中根据iscompactor条件构建每一层
如果iscompactor 为 true，需要为Conv和C3模块传入iscompactor参数，cur_conv_idx根据模块中的卷积层数计算后更新

## common.py 
### class Conv
Conv是yolo中最基础的模块，其中有卷积，BN，激活层，需要做的是根据iscompactor条件把compactor添加在BN层之后
将cur_conv_idx与iscompactor记录在该类中
### class Bottleneck
根据iscompactor，为cv1添加参数
### class C3
根据条件，为cv1和cv2添加参数  todo cv3也可以添加compactor层，除了SPPF前的那个C3
### class SPPF
仅接收cur_conv_idx，为其中的卷积添加id

## train_resYolo.py
在train.py基础上进行修改
在Backward之后，Optimize之前，需要为为compactor增加L2正则惩罚项梯度
在epoch全部结束后，调用compactor_convert压缩模型，并保存

# 新增记录
## compactor.py
压缩层定义
## constants.py
用于剪枝的yolo5s_succeeding记录，仅用于yolov5s结构
## resrep_convert.py
用于将模型中的卷积层与BN层合并，并进行实际的剪枝，保存剪枝后的模型
## resrep_util.py 
对模型中的compactor进行一些操作

# todo
- 训练中对mask的修改
- 增加C3模块中对CV3的剪枝