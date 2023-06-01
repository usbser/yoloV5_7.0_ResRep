import torch
import torch.nn.init as init
from torch.nn import Conv2d
import numpy as np

class CompactorLayer(torch.nn.Module):

    def __init__(self, num_features, conv_idx, ):
        super(CompactorLayer, self).__init__()
        self.conv_idx = conv_idx
        # 创建一个1x1的卷积层，作为压缩器的一部分。
        self.pwc = Conv2d(in_channels=num_features, out_channels=num_features, kernel_size=1,
                          stride=1, padding=0, bias=False)
        #创建一个大小为in_channels的单位矩阵。
        identity_mat = np.eye(num_features, dtype=np.float32)
        #将单位矩阵作为卷积层的权重，并复制给卷积层的权重。
        self.pwc.weight.data.copy_(torch.from_numpy(identity_mat).reshape(num_features, num_features, 1, 1))
        #注册一个缓冲区变量mask，并初始化为一个全1的张量。
        self.register_buffer('mask', torch.ones(num_features))
        #初始化mask为全1的张量
        init.ones_(self.mask)
        self.num_features = num_features

    def forward(self, inputs):
        return self.pwc(inputs)

    # 将mask对应zero_indices位置的元素置为0。 zero_indices是一个list，记录了要被搞掉的卷积核号
    def set_mask(self, zero_indices):
        #创建一个大小为num_features的全1的NumPy数组。
        new_mask_value = np.ones(self.num_features, dtype=np.float32)
        #对应位置的元素置为0。
        new_mask_value[np.array(zero_indices)] = 0.0
        self.mask.data = torch.from_numpy(new_mask_value).cuda().type(torch.cuda.FloatTensor)

    def set_weight_zero(self, zero_indices):
        new_compactor_value = self.pwc.weight.data.detach().cpu().numpy()
        new_compactor_value[np.array(zero_indices), :, :, :] = 0.0
        self.pwc.weight.data = torch.from_numpy(new_compactor_value).cuda().type(torch.cuda.FloatTensor)

    #获取mask中数值为1的元素数量的函数。
    def get_num_mask_ones(self):
        mask_value = self.mask.cpu().numpy()
        return np.sum(mask_value == 1)

    #获取mask中数值为1的元素占总元素数量的比例的函数。
    def get_remain_ratio(self):
        return self.get_num_mask_ones() / self.num_features

    #定义了获取卷积层权重的副本的函数。
    def get_pwc_kernel_detach(self):
        return self.pwc.weight.detach()

    def get_lasso_vector(self):
        lasso_vector = torch.sqrt(torch.sum(self.get_pwc_kernel_detach() ** 2, dim=(1, 2, 3))).cpu().numpy()
        return lasso_vector

    def get_metric_vector(self):
        # 计算L2向量，对卷积层权重的每个通道计算其在空间维度上的平方和，
        # 然后进行开方运算，并将结果转换为NumPy数组。
        metric_vector = torch.sqrt(torch.sum(self.get_pwc_kernel_detach() ** 2, dim=(1, 2, 3))).cpu().numpy()
        return metric_vector