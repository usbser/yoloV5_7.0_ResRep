import torch.nn as nn
import numpy as np
from collections import defaultdict

# 得到compator每一层mask等于1的数量，compator每一个卷积核的l2范数
#   pacesetter is not included here
def resrep_get_layer_mask_ones_and_metric_dict(model:nn.Module):
    # 存储各层掩码中值为1的数量key = 层号，value=这一层mask为1的数量  {1：16  ，3：16，....}
    layer_mask_ones = {}
    # 用来存储各层l2向量key =（层号，卷积核号）， value=卷积核对应的L2范数 #{(1, 0): 1.2030473, (1, 1): 1.3146878, (1, 2): 0.981252 ...}
    layer_metric_dict = {}
    # 存储每一层compator的宽度信息，layer_mask_ones的value
    report_deps = []

    # 遍历model的所有子模块
    for child_module in model.modules():
        # 针对compator层
        if hasattr(child_module, 'conv_idx'):
            # 获取当前子模块掩码中值为1的数量，并将其存储到字典layer_mask_ones中
            layer_mask_ones[child_module.conv_idx] = child_module.get_num_mask_ones()
            # 计算compator层卷积核权重的L2范数
            metric_vector = child_module.get_metric_vector()
            # 遍历当前层卷积核权重的L2范数中的每一个元素
            for i in range(len(metric_vector)):
                # 将当前子模块的L2范数存储到字典layer_metric_dict中
                layer_metric_dict[(child_module.conv_idx, i)] = metric_vector[i]
            # 将当前子模块掩码中值为1的数量添加到列表report_deps中
            report_deps.append(layer_mask_ones[child_module.conv_idx])
    # print('now active deps: ', report_deps)
    return layer_mask_ones, layer_metric_dict

# 根据每一层被减掉的卷积核记录layer_masked_out_filters，更新compator的mask
def set_model_masks(model, layer_masked_out_filters):
    for child_module in model.modules():
        # 处理每一个在layer_masked_out_filters记录的compator层
        if hasattr(child_module, 'conv_idx') and child_module.conv_idx in layer_masked_out_filters:
            # 将compator层中的mask更新为layer_masked_out_filters中对应的值
            child_module.set_mask(layer_masked_out_filters[child_module.conv_idx])

# 根据compator每一层的mask获得当前的宽度信息，以及得到当前每一个卷积核的l2范数metric_dict
def resrep_get_deps_and_metric_dict(origin_deps, model:nn.Module, pacesetter_dict):
    #更新后的compator宽度信息
    new_deps = np.array(origin_deps)
    # 获取各层掩码中值为1的数量layer_ones 和 每一个卷积核的L2范数
    layer_ones, metric_dict = resrep_get_layer_mask_ones_and_metric_dict(model)
    # 遍历各层掩码中值为1的数量
    for idx, ones in layer_ones.items():
        assert ones <= origin_deps[idx]
        # 用mask更新这一compator层的宽度信息
        new_deps[idx] = ones
        # TODO include pacesetter_dict here
        # 如果pacesetter_dict不为空，
        if pacesetter_dict is not None:
            # 在pacesetter_dict中查找后继层
            for follower, pacesetter in pacesetter_dict.items():

                if follower != pacesetter and pacesetter == idx:
                    new_deps[follower] = ones

    return new_deps, metric_dict

# 计算总共剪掉了多少个卷积核
def get_cur_num_deactivated_filters(origin_deps, cur_deps, follow_dict):
    assert len(origin_deps) == len(cur_deps)
    # 计算每一层被减少了多少个卷积核
    diff = origin_deps - cur_deps
    assert np.sum(diff < 0) == 0

    result = 0
    # 遍历origin_deps中的每一个元素
    for i in range(len(origin_deps)):
        # 如果这一层有后继记录在follow_dict中，就算了
        if (follow_dict is not None) and (i in follow_dict) and (follow_dict[i] != i):
            pass
        else:
            # 否则，将origin_deps[i]和cur_deps[i]的差值加到result上
            result += origin_deps[i] - cur_deps[i]
    return result


def resrep_mask_model(origin_deps, model:nn.Module):
    #根据compator每一层的mask获得当前的宽度信息，以及得到当前每一个卷积核的l2范数metric_dict
    cur_deps, metric_dict = resrep_get_deps_and_metric_dict(origin_deps, model)

    # 根据l2范数的大小，从小到大来排序，返回字典中的key，即（层号，卷积核号）的元组 ，
    sorted_metric_dict = sorted(metric_dict, key=metric_dict.get)

    next_deactivated_max = 4
    # 尝试进行剪枝
    attempt_deps = np.array(origin_deps)
    i = 0  # 用于取出sorted_metric_dict前next_deactivated_max个卷积核，进行尝试剪枝
    skip_idx = []
    #通过更改deps尝试减少过滤器的数量。
    # 从sorted_metric_dict中获取当前索引i对应的卷积核（层号，卷积核号）（attempt_layer_filter）。
    # 如果该卷积层的过滤器数量小于等于num_at_least，表示已经达到了最小数量要求，
    #   将该索引添加到skip_idx列表中，该层不再被剪枝
    #   继续下一次循环。
    # 如果不满足以上条件，则将当前卷积层的过滤器数量减少1。
    while True:
        attempt_layer_filter = sorted_metric_dict[i]
        if attempt_deps[attempt_layer_filter[0]] <= 20: # num_at_least = 1，
            skip_idx.append(i)    #表示的这一次的剪枝要取消掉，在下面进行实际剪枝的时候跳过这一次
            i += 1
            continue
        attempt_deps[attempt_layer_filter[0]] -= 1   # 给这一层进行剪枝
        # TODO include pace_setter dict here
        i += 1
        if i >= next_deactivated_max:
            break

    # 记录每个卷积层中需要置零的权重位置。key = 层号, value = 卷积核号的list
    layer_masked_out_filters = defaultdict(list)  # layer_idx : [zeros]
    # i的值表示在前面的压缩过程中被剪枝的卷积层的数量。
    for k in range(i):
        #检查当前循环索引k是否在skip_idx列表中，如果不在，则表示该卷积层不是被跳过的卷积层，需要进行压缩操作。
        if k not in skip_idx:
            #将当前卷积层的索引和权重位置添加到layer_masked_out_filters字典中。
            # sorted_metric_dict[k][0]表示当前卷积层的索引，
            # sorted_metric_dict[k][1]表示需要置零的权重位置。
            # 这样，通过索引可以找到对应的卷积层，并将需要置零的权重位置添加到相应的列表中。
            layer_masked_out_filters[sorted_metric_dict[k][0]].append(sorted_metric_dict[k][1])
    # 更新模型中compator中的mask信息，将对应层的对应卷积核的对应mask置0
    set_model_masks(model, layer_masked_out_filters)


def get_compactor_mask_dict(model:nn.Module):
    # 保存模型缓冲区中'compactor层的mask'
    compactor_name_to_mask = {}
    # 保存模型参数'compactor层的weight'
    compactor_name_to_kernel_param = {}

    # 遍历模型的所有缓冲区,BN层的running_mean/var,自定义的compactor.mask
    test_buffer = {}
    for name, buffer in model.named_buffers():
        test_buffer[name] = buffer
        # 如果缓冲区的名字中包含'compactor.mask'
        if 'compactor.mask' in name:
            # 将这个缓冲区存入compactor_name_to_mask字典中，键为缓冲区名字去除'mask'后的部分，值为该缓冲区
            compactor_name_to_mask[name.replace('mask', '')] = buffer

    test_state_dict = {}
    for k, v in model.state_dict().items():
        test_state_dict[k] = v

    # 遍历模型的所有参数
    test_named_parameters = {}
    for name, param in model.named_parameters():
        test_named_parameters[name] = buffer
        # 如果参数的名字中包含'compactor.pwc.weight'
        if 'compactor.pwc.weight' in name:
            # 将这个参数存入compactor_name_to_kernel_param字典中，键为参数名字去除'pwc.weight'后的部分，值为该参数
            compactor_name_to_kernel_param[name.replace('pwc.weight', '')] = param
    # key = 这一层compator卷积核，value = 这一层compator的mask
    result = {}  # 保存处理后的结果
    # 遍历compactor层的weight
    for name, kernel in compactor_name_to_kernel_param.items():
        # 获取与当前kernel对应的mask
        mask = compactor_name_to_mask[name]
        # 获取mask中元素的数量
        num_filters = mask.nelement()

        # 如果kernel的维度是4(卷积)
        if kernel.ndimension() == 4:
            # 如果mask的维度是1
            if mask.ndimension() == 1:
                # 将mask变形为[num_filters, 1]，然后复制num_filters次，
                # 得到一个[num_filters, num_filters]的broadcast_mask
                broadcast_mask = mask.reshape(-1, 1).repeat(1, num_filters)
                # 将broadcast_mask变形为[num_filters, num_filters, 1, 1]，
                # 然后保存到result字典中，键为kernel，值为变形后的broadcast_mask
                result[kernel] = broadcast_mask.reshape(num_filters, num_filters, 1, 1)
            else:
                # 如果mask的维度不是1，那么它应该是4
                assert mask.ndimension() == 4
                # 将mask保存到result字典中，键为kernel，值为mask
                result[kernel] = mask
        # else:
            # 将mask保存到result字典中，键为kernel，值为mask
            # assert kernel.ndimension() == 1
            # 将mask保存到result字典中，键为kernel，值为mask
            # result[kernel] = mask
    return result


# 尝试计算当前进行剪枝后每一层的深度
def get_deps_if_prune_low_metric(origin_deps, model, threshold, pacesetter_dict):

    cur_deps = np.array(origin_deps)
    # 遍历model的所有子模块
    for child_module in model.modules():
        # 处理其中的compator层
        if hasattr(child_module, 'conv_idx'):
            # 获取当前compator层的卷积核的L2范数
            metric_vector = child_module.get_metric_vector()
            # 计算度量向量中小于阈值的元素的数量
            num_filters_under_thres = np.sum(metric_vector < threshold)
            # 从cur_deps数组的对应位置减去小于阈值的元素的数量
            cur_deps[child_module.conv_idx] -= num_filters_under_thres
            # 确保cur_deps数组的对应位置的值不小于1
            cur_deps[child_module.conv_idx] = max(1, cur_deps[child_module.conv_idx])   #TODO
            # TODO pacesetter?
            # 如果pacesetter_dict参数不为空
            if pacesetter_dict is not None:
                # 遍历pacesetter_dict字典
                for follower, pacesetter in pacesetter_dict.items():
                    # 如果pacesetter的值等于当前子模块的conv_idx，并且pacesetter不等于follower
                    if pacesetter == child_module.conv_idx and pacesetter != follower:
                        # 从cur_deps数组的对应位置减去小于阈值的元素的数量
                        cur_deps[follower] -= num_filters_under_thres
                        # 确保cur_deps数组的对应位置的值不小于1
                        cur_deps[follower] = max(1, cur_deps[follower])
    return cur_deps


def resrep_get_unmasked_deps(origin_deps, model:nn.Module, pacesetter_dict):
    # 获取模型每一层中未被掩码的宽度
    unmasked_deps = np.array(origin_deps)
    # 遍历model的所有子模块
    for child_module in model.modules():
        # 如果当前子模块有属性conv_idx
        if hasattr(child_module, 'conv_idx'):
            # 获取当前子模块mask中值为1的数量
            layer_ones = child_module.get_num_mask_ones()
            # 更新unmasked_deps数组的对应位置的值
            unmasked_deps[child_module.conv_idx] = layer_ones

            # 如果pacesetter_dict参数不为空
            if pacesetter_dict is not None:
                # 遍历pacesetter_dict字典
                for follower, pacesetter in pacesetter_dict.items():
                    # 如果pacesetter的值等于当前子模块的conv_idx
                    if pacesetter == child_module.conv_idx:
                        # 更新unmasked_deps数组的对应位置的值
                        unmasked_deps[follower] = layer_ones
                    # print('cur conv ', child_module.conv_idx, 'follower is ', follower)
    return unmasked_deps

