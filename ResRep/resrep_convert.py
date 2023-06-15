import os

from torch import nn


import numpy as np
import torch.nn.functional as F
import torch

import sys
sys.path.append("..")
from ResRep.misc import save_hdf5
from models.common import Conv
from copy import deepcopy
from utils.torch_utils import de_parallel

# 该函数用于融合卷积核和BN层的权重。
# 将卷积核的权重通过BN层的参数进行调整，得到融合后的卷积核权重。
def _fuse_kernel(kernel, gamma, running_var, eps):
    print('fusing: kernel shape', kernel.shape)
    # 卷积核和BN层的权重融合
    std = np.sqrt(running_var + eps)
    t = gamma / std
    t = np.reshape(t, (-1, 1, 1, 1))
    print('fusing: t', t.shape)
    t = np.tile(t, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    return kernel * t # 返回融合后的卷积核权重

# 该函数用于融合卷积层和BN层的偏置。
# 将卷积层的偏置通过BN层的参数进行调整，得到融合后的偏置
def _fuse_bias(running_mean, running_var, gamma, beta, eps, bias=None):
    # 卷积层和BN层的偏置融合
    if bias is None:
        return beta - running_mean * gamma / np.sqrt(running_var + eps)
    else:
        return beta + (bias - running_mean) * gamma / np.sqrt(running_var + eps)

# 该函数用于找到卷积层和对应的BN层的参数，
# 然后调用上面两个函数进行融合。
def fuse_conv_bn(save_dict, pop_name_set, kernel_name):
    # 该函数用于融合卷积层和对应的BN层的参数。
    # 首先，根据卷积层的名称找到BN层的参数名称。
    mean_name = kernel_name.replace('.conv.weight', '.bn.running_mean')
    var_name = kernel_name.replace('.conv.weight', '.bn.running_var')
    gamma_name = kernel_name.replace('.conv.weight', '.bn.weight')
    beta_name = kernel_name.replace('.conv.weight', '.bn.bias')
    # 将这些参数名称添加到待删除集合中。
    pop_name_set.add(mean_name)
    pop_name_set.add(var_name)
    pop_name_set.add(gamma_name)
    pop_name_set.add(beta_name)
    # 获取批量归一化层的参数。
    mean = save_dict[mean_name]
    var = save_dict[var_name]
    gamma = save_dict[gamma_name]
    beta = save_dict[beta_name]
    # 获取卷积层的权重。
    kernel_value = save_dict[kernel_name]
    print('kernel name', kernel_name)
    print('kernel, mean, var, gamma, beta', kernel_value.shape, mean.shape, var.shape, gamma.shape, beta.shape)
    # 调用前面定义的两个函数进行融合，并返回融合后的卷积核权重和偏置。
    return _fuse_kernel(kernel_value, gamma, var, eps=1e-5), _fuse_bias(mean, var, gamma, beta, eps=1e-5)

# 该函数用于将融合后的卷积核和压缩层的权重进行卷积操作，
# 得到压缩后的卷积核和偏置
def fold_conv(fused_k, fused_b, thresh, compactor_mat):
    # 该函数用于将融合后的卷积核和压缩层的权重进行卷积操作，得到压缩后的卷积核和偏置。
    # 首先，计算压缩层的权重的L2范数，并找到范数小于阈值的索引。
    metric_vec = np.sqrt(np.sum(compactor_mat ** 2, axis=(1, 2, 3)))
    filter_ids_below_thresh = np.where(metric_vec < thresh)[0]
    # 如果所有的卷积核的范数都小于阈值，保留范数最大的卷积核。
    if len(filter_ids_below_thresh) == len(metric_vec):
        sortd_ids = np.argsort(metric_vec)
        filter_ids_below_thresh = sortd_ids[:-1]    #TODO preserve at least one filter
    # 删除范数小于阈值的卷积核。
    # filter_ids_below_thresh = [1,2,3,4,5]  # todo 用于测试
    if len(filter_ids_below_thresh) > 0:
        compactor_mat = np.delete(compactor_mat, filter_ids_below_thresh, axis=0)
    # 将融合后的卷积核和压缩层的权重进行卷积操作，得到压缩后的卷积核。# out_channels代表卷积核的数量，
    kernel = F.conv2d(torch.from_numpy(fused_k).cuda().permute(1, 0, 2, 3), torch.from_numpy(compactor_mat).cuda(),
                      padding=(0, 0)).permute(1, 0, 2, 3)

    # 计算压缩后的偏置。
    Dprime = compactor_mat.shape[0]
    bias = np.zeros(Dprime)
    # 对于每一个压缩后的卷积核，计算其偏置。
    for i in range(Dprime):
        bias[i] = fused_b.dot(compactor_mat[i,:,0,0])
    # 如果偏置不是一个numpy数组，将其转换为numpy数组。
    if type(bias) is not np.ndarray:
        bias = np.array([bias])
    # 返回压缩后的卷积核、偏置以及被删除的卷积核的索引。
    return kernel, bias, filter_ids_below_thresh


def compactor_convert(model, thresh, succ_strategy, save_path):
    origin_deps = []
    for submodule in model.modules():
        if isinstance(submodule, Conv):
            origin_deps.append(submodule.conv.out_channels)
    # 存储每个压缩层的权重，key = 层号conv_idx , value = 权重weight
    compactor_mats = {}
    for submodule in model.modules():
        if hasattr(submodule, 'conv_idx'):
            compactor_mats[submodule.conv_idx] = submodule.pwc.weight.detach().cpu().numpy()
    # 用于保存最终每一层剩下的宽度（卷积核数）
    pruned_deps = np.array(origin_deps)
    # 当前正在被处理的compator层
    cur_conv_idx = -1
    pop_name_set = set() # 待删除的参数的集合（融合掉的BN层）
    # 创建一个列表来保存所有卷积核的名称
    kernel_name_list = []
    save_dict = {}
    # 遍历模型的所有参数（所有BasicBranch中的两个卷积conv1和conv2），将所有参数值保存到字典save_dict中，将除了compactor外的卷积核和最终线性层的名称添加到列表kernel_name_list中
    for k, v in model.state_dict().items():
        v = v.detach().cpu().numpy()
        if v.ndim == 4 and 'compactor.pwc' not in k : #and 'align_opr.pwc' not in k: # align_opr是直接映射层，无作用
            kernel_name_list.append(k)
        save_dict[k] = v
    kernel_name_list = kernel_name_list[0:-3] # 去掉detece的三个卷积
    # 根据kernel_name_list遍历所有卷积核
    for conv_id, kernel_name in enumerate(kernel_name_list):
        kernel_value = save_dict[kernel_name]
        # 如果是全连接层，则跳过
        if kernel_value.ndim == 2:
            continue
        # 将卷积层和其后的BN层融合
        fused_k, fused_b = fuse_conv_bn(save_dict, pop_name_set, kernel_name)
        cur_conv_idx += 1
        fold_direct = cur_conv_idx in compactor_mats
        # 如果当前卷积层是一个直接压缩层或者是一个追随层
        if fold_direct :
            fm = compactor_mats[cur_conv_idx]
            # 使用压缩层的权重矩阵来裁剪卷积核
            fused_k, fused_b, pruned_ids = fold_conv(fused_k, fused_b, thresh, fm)
            pruned_deps[cur_conv_idx] -= len(pruned_ids)
            print('pruned ids: ', pruned_ids)
            # 当前层被剪了，同时当前卷积层有后续层（调整后续层）
            if len(pruned_ids) > 0 and conv_id in succ_strategy:
                followers = succ_strategy[conv_id]
                if type(followers) is not list:
                    followers = [followers]
                # 对每个后续层进行处理
                for fo in followers:
                    fo_kernel_name = kernel_name_list[fo]
                    fo_value = save_dict[fo_kernel_name]
                    # 如果是卷积层，直接在输入通道维度上删除对应的卷积核
                    if fo_value.ndim == 4:  #
                        fo_value = np.delete(fo_value, pruned_ids, axis=1)
                    save_dict[fo_kernel_name] = fo_value
        # 更新卷积层的权重和偏置
        save_dict[kernel_name] = fused_k
        save_dict[kernel_name.replace('.weight', '.bias')] = fused_b
    # 将裁剪后的每一层的宽度信息（list）添加到字典中
    # save_dict['deps'] = pruned_deps
    # 删除所有融合后的BN层的参数
    for name in pop_name_set:
        save_dict.pop(name)
    # 删除所有的BN层和压缩层的参数，并将参数名称的前缀"module."去掉
    final_dict = {}
    for k, v in save_dict.items():
        if 'num_batches' not in k and 'compactor' not in k:
            v = torch.from_numpy(v) if isinstance(v,np.ndarray) else v
            final_dict[k.replace('module.', '')] = v
        # 将裁剪后的模型参数保存为HDF5文件

    #newmodel = deepcopy(de_parallel(ckpt['model'])).half()
    model.cuda()
    for param in model.parameters():
        param.data = param.data.float()
    if int(os.getenv('RANK', -1)) != -1:
        model.module.fuse()
        namemodel = model.module.named_modules()
    else:
        model.fuse()
        namemodel = model.named_modules()
    ckpt = {}
    for name, submodule in namemodel:
        if isinstance(submodule, Conv):
            if hasattr(submodule, 'compactor'):
                delattr(submodule, 'compactor')
            out_ch = final_dict[name+'.conv.weight'].size(0)
            in_ch = final_dict[name+'.conv.weight'].size(1)
            submodule.conv = nn.Conv2d(in_ch,out_ch,submodule.conv.kernel_size,submodule.conv.stride,padding=submodule.conv.padding,groups=submodule.conv.groups,dilation=submodule.conv.dilation)

    model.load_state_dict(final_dict, strict=False)
    ckpt['model'] = model
    ckpt['final_dict'] = final_dict
    ckpt['pruned_deps'] = pruned_deps
    torch.save(ckpt, save_path)
    print('---------------saved {} numpy arrays to {}---------------'.format(len(save_dict), save_path))


def compactor_convert_mi1(model, origin_deps, thresh, save_path):
    compactor_mats = {}
    for submodule in model.modules():
        if hasattr(submodule, 'conv_idx'):
            compactor_mats[submodule.conv_idx] = submodule.pwc.weight.detach().cpu().numpy()

    pruned_deps = np.array(origin_deps)

    pop_name_set = set()

    kernel_name_list = []
    save_dict = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu().numpy()
        if v.ndim in [2, 4] and 'compactor.pwc' not in k and 'align_opr.pwc' not in k:
            kernel_name_list.append(k)
        save_dict[k] = v

    for conv_id, kernel_name in enumerate(kernel_name_list):
        kernel_value = save_dict[kernel_name]
        if kernel_value.ndim == 2:
            continue
        fused_k, fused_b = fuse_conv_bn(save_dict, pop_name_set, kernel_name)

        save_dict[kernel_name] = fused_k
        save_dict[kernel_name.replace('.weight', '.bias')] = fused_b

        fold_direct = conv_id in compactor_mats
        if fold_direct:
            fm = compactor_mats[conv_id]
            fused_k, fused_b, pruned_ids = fold_conv(fused_k, fused_b, thresh, fm)

            save_dict[kernel_name] = fused_k
            save_dict[kernel_name.replace('.weight', '.bias')] = fused_b

            pruned_deps[conv_id] -= len(pruned_ids)
            print('pruned ids: ', pruned_ids)
            if len(pruned_ids) == 0:
                continue
            fo_kernel_name = kernel_name_list[conv_id + 1]
            if 'linear' in fo_kernel_name:
                fo_value = save_dict[fo_kernel_name]
                fc_idx_to_delete = []
                num_filters = kernel_value.shape[0]
                fc_neurons_per_conv_kernel = fo_value.shape[1] // num_filters
                print('{} filters, {} neurons per kernel'.format(num_filters, fc_neurons_per_conv_kernel))
                base = np.arange(0, fc_neurons_per_conv_kernel * num_filters, num_filters)
                for i in pruned_ids:
                    fc_idx_to_delete.append(base + i)
                if len(fc_idx_to_delete) > 0:
                    fo_value = np.delete(fo_value, np.concatenate(fc_idx_to_delete, axis=0), axis=1)
                save_dict[fo_kernel_name] = fo_value
            else:
                #   this_layer - following_pw - following_dw
                #   adjust the beta of following_dw by the to-be-deleted channel of following_pw
                #   delete the to-be-deleted channel of following_pw
                fol_dw_kernel_name = kernel_name_list[conv_id + 1]
                fol_dw_kernel_value = save_dict[fol_dw_kernel_name]
                fol_dw_beta_name = fol_dw_kernel_name.replace('conv.weight', 'bn.bias')
                fol_dw_beta_value = save_dict[fol_dw_beta_name]

                pw_kernel_name = kernel_name_list[conv_id + 2]
                pw_kernel_value = save_dict[pw_kernel_name]
                pw_beta_name = pw_kernel_name.replace('conv.weight', 'bn.bias')
                pw_beta_value = save_dict[pw_beta_name]
                pw_var_value = save_dict[pw_kernel_name.replace('conv.weight', 'bn.running_var')]
                pw_gamma_value = save_dict[pw_kernel_name.replace('conv.weight', 'bn.weight')]

                for pri in pruned_ids:
                    compensate_beta = np.abs(fol_dw_beta_value[pri]) * (pw_kernel_value[:, pri, 0, 0] * pw_gamma_value / np.sqrt(pw_var_value + 1e-5))  # TODO because of relu
                    pw_beta_value += compensate_beta
                save_dict[pw_beta_name] = pw_beta_value
                save_dict[pw_kernel_name] = np.delete(pw_kernel_value, pruned_ids, axis=1)

                fol_dw_kernel_value = np.delete(fol_dw_kernel_value, pruned_ids, axis=0)
                fol_dw_beta_value = np.delete(fol_dw_beta_value, pruned_ids)
                save_dict[fol_dw_kernel_name] = fol_dw_kernel_value
                save_dict[fol_dw_beta_name] = fol_dw_beta_value
                fol_dw_gamma_name = fol_dw_kernel_name.replace('conv.weight', 'bn.weight')
                fol_dw_mean_name = fol_dw_kernel_name.replace('conv.weight', 'bn.running_mean')
                fol_dw_var_name = fol_dw_kernel_name.replace('conv.weight', 'bn.running_var')
                save_dict[fol_dw_gamma_name] = np.delete(save_dict[fol_dw_gamma_name], pruned_ids)
                save_dict[fol_dw_mean_name] = np.delete(save_dict[fol_dw_mean_name], pruned_ids)
                save_dict[fol_dw_var_name] = np.delete(save_dict[fol_dw_var_name], pruned_ids)
                pruned_deps[conv_id+1] -= len(pruned_ids)

    save_dict['deps'] = pruned_deps
    for name in pop_name_set:
        save_dict.pop(name)

    final_dict = {k.replace('module.', '') : v for k, v in save_dict.items() if 'num_batches' not in k and 'compactor' not in k}

    save_hdf5(final_dict, save_path)
    print('---------------saved {} numpy arrays to {}---------------'.format(len(save_dict), save_path))