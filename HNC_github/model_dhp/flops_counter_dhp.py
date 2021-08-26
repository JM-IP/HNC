"""
Reference
https://github.com/sovrasov/flops-counter.pytorch.git
"""
import torch
import torch.nn as nn
import numpy as np
from model_dhp.dhp_base import conv_dhp
# from IPython import embed
import model_dhp.parametric_quantization as PQ
from torch.autograd import Variable
import re


# Function to extract all the numbers from the given string
def getNumbers(str):
    array = re.findall(r'[0-9]+', str)
    return array

def set_output_dimension(model, input_res):
    assert type(input_res) is tuple, 'Please provide the size of the input image.'
    assert len(input_res) >= 3, 'Input image should have 3 dimensions.'
    feat_model = add_feature_dimension(model)
    feat_model.eval().start_dimension_add()
    device = list(feat_model.parameters())[-1].device
    batch = torch.FloatTensor(1, *input_res).to(device)
    _ = feat_model(batch)
    feat_model.stop_dimension_add()


def get_flops_prune_only(model, cfg, pruned=True):
    return get_flops_grad(model, cfg, init_flops=True, pruned=pruned).item()


def get_flops(model, cfg, init_flops=False, pruned=True):
    return get_flops_grad(model, cfg, init_flops=init_flops, pruned=pruned).item()


def get_flops_grad(model, cfg, init_flops=False, pruned=True):
    flops = 0
    n_list = []
    _, _, n_dict = PQ.network_size_activations(model, cfg)
    if not init_flops:
        if model.args.base == 'MobileNet' or model.args.base == 'MobileNetV2':
            act_n = 32
            for name, module in model.named_modules():
                if is_supported_instance(module):
                    if isinstance(module, (conv_dhp)):
                        w_n = PQ.get_percision(module.quantize_w, cfg=cfg)
                        n_list.append(act_n * w_n)
                        act_n_name = name + '.a_quant.a_size'
                        act_n = n_dict[act_n_name]
                    if isinstance(module, (nn.Linear)):
                        try:
                            w_n = PQ.get_percision(module.quantize_w, cfg=cfg)
                        except AttributeError:
                            w_n = 32
                        n_list.append(act_n * w_n)
        elif model.args.base == 'VGG':
            act_n = 32
            for name, module in model.named_modules():
                if is_supported_instance(module):
                    if isinstance(module, (conv_dhp)):
                        w_n = PQ.get_percision(module.quantize_w, cfg=cfg)
                        n_list.append(act_n * w_n)
                        act_n_name = name + '.a_quant.a_size'
                        act_n = n_dict[act_n_name]
                    if isinstance(module, (nn.Linear)):
                        w_n = PQ.get_percision(module.quantize_w, cfg=cfg)
                        n_list.append(act_n * w_n)
        elif  '18' in model.args.base:
            for name, module in model.named_modules():
                if is_supported_instance(module):

                    if isinstance(module, (conv_dhp)):
                        if '0' in name:
                            n_list.append(PQ.get_percision(module.quantize_w, cfg=cfg)*32)
                        elif 'layer1' in name or 'downsample' in name:
                            act_n_name = name.replace('.layer1', '.act_out.a_size')
                            act_n_name = act_n_name.replace('.downsample', '.act_out.a_size')
                            s = int(getNumbers(act_n_name)[0])
                            # [int(s) for s in act_n_name.split() if s.isdigit()]
                            if s == 1:
                                act_n_name = act_n_name.replace('act_out', 'a_quant')
                            act_n_name = act_n_name.replace(str(s), str(s-1))
                            act_n = n_dict[act_n_name]
                            n_list.append(PQ.get_percision(module.quantize_w, cfg=cfg) * act_n)
                        elif 'layer2' in name:
                            act_n_name = name.replace('.layer2', '.layer1.a_quant.a_size')
                            act_n = n_dict[act_n_name]
                            n_list.append(PQ.get_percision(module.quantize_w, cfg=cfg) * act_n)

                    elif isinstance(module, (nn.Linear)):
                        try:
                            w_n = PQ.get_percision(module.quantize_w, cfg=cfg)
                        except AttributeError:
                            w_n = 32
                        n_list.append(n_dict['features.8.act_out.a_size'] * w_n)

        else:
            for name, module in model.named_modules():
                if is_supported_instance(module):

                    if isinstance(module, (conv_dhp)):
                        if '0' in name:
                            n_list.append(PQ.get_percision(module.quantize_w, cfg=cfg)*32)
                        elif 'layer1' in name or 'downsample' in name:
                            act_n_name = name.replace('.layer1', '.act_out.a_size')
                            act_n_name = act_n_name.replace('.downsample', '.act_out.a_size')
                            s = int(getNumbers(act_n_name)[0])
                            # [int(s) for s in act_n_name.split() if s.isdigit()]
                            if s == 1:
                                act_n_name = act_n_name.replace('act_out', 'a_quant')
                            act_n_name = act_n_name.replace(str(s), str(s-1))
                            act_n = n_dict[act_n_name]
                            n_list.append(PQ.get_percision(module.quantize_w, cfg=cfg) * act_n)
                        elif 'layer2' in name:
                            act_n_name = name.replace('.layer2', '.layer1.a_quant.a_size')
                            act_n = n_dict[act_n_name]
                            n_list.append(PQ.get_percision(module.quantize_w, cfg=cfg) * act_n)

                    elif isinstance(module, (nn.Linear)):
                        try:
                            w_n = PQ.get_percision(module.quantize_w, cfg=cfg)
                        except AttributeError:
                            w_n = 32
                        n_list.append(32 * w_n)
                        # n_list.append(PQ.get_percision(module.quantize_w, cfg=cfg)*n_dict['features.9.act_out.a_size'])
                    # for i in n_list:
                    #     if i.item()>32*32:
                    #         raise('yjm is here',i.item())
                            # print('yjm is here',i.item())
    i = 0
    for name, module in model.named_modules():
        if is_supported_instance(module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, conv_dhp)):
                if not init_flops:
                    flops += conv_calc_flops_grad(module, n_list[i])
                else:
                    flops += conv_calc_flops_grad(module, 32 * 32, pruned)
                i += 1
                # TODO: shortcut cal is need to be fixed!
            elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
                flops += relu_calc_flops(module) * 32
                # if isinstance(module, nn.ReLU):
                #     print(module)
            elif isinstance(module, (nn.Linear)):
                if not init_flops:
                    flops += linear_calc_flops(module, n_list[i])
                else:
                    flops += linear_calc_flops(module, 32 * 32)
                i += 1
            elif isinstance(module, (nn.BatchNorm2d)):
                flops += bn_calc_flops(module) * 32 * 32
    return flops

def conv_calc_flops_grad(self, nn, pruned=True, model=None):
    # Do not count bias addition
    # batch_size = 1
    output_dims = np.prod(self.__output_dims__)

    kernel_dims = np.prod(self.kernel_size) if isinstance(self.kernel_size, tuple) else self.kernel_size ** 2
    in_channels = self.in_channels_remain if hasattr(self, 'in_channels_remain') else self.in_channels
    out_channels = self.out_channels_remain if hasattr(self, 'out_channels_remain') else self.out_channels
    groups = self.groups_remain if hasattr(self, 'groups_remain') else self.groups
    # groups = self.groups
    if pruned:
        in_channels_num = self.in_channels_num
        out_channels_num = self.out_channels_num
    else:
        in_channels_num = in_channels
        out_channels_num = out_channels
    # if pruned:
    #     if not in_channels == in_channels_num:
    #         print(self)
    #         print('in_channels is ', in_channels, 'in_channels_num is ', in_channels_num)
    #         # raise()
    #     if not out_channels == out_channels_num:
    #         print(self)
    #         print('out_channels is ', out_channels, 'out_channels_num is ', out_channels_num)
    #         # raise()
    filters_per_channel = out_channels_num / groups
    conv_per_position_flops = kernel_dims * in_channels_num * filters_per_channel

    active_elements_count = output_dims

    overall_conv_flops = conv_per_position_flops * active_elements_count
    return overall_conv_flops*nn


def get_parameters_prune_only(model, cfg):
    return get_parameters(model, cfg, init_params=True)


def get_parameters(model, cfg, init_params=False):
    parameters = 0
    for module in model.modules():
        if is_supported_instance(module):
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
                for p in module.parameters():
                    parameters += p.nelement() * 32
            elif isinstance(module, nn.Linear):
                in_features = module.in_features_remain if hasattr(module, 'in_features_remain') else module.in_features
                out_features = module.out_features_remain if hasattr(module, 'out_features_remain') else module.out_features
                try:
                    n = PQ.get_percision(module.quantize_w, cfg).item()
                except AttributeError:
                    n = 32
                if not init_params:
                    parameters += in_features * out_features * n
                else:
                    parameters += in_features * out_features * 32
                if module.bias is not None:
                    if not init_params:
                        parameters += module.out_features * n
                    else:
                        parameters += in_features * out_features * 32
            elif isinstance(module, (conv_dhp)):
                in_channels = module.in_channels_remain if hasattr(module, 'in_channels_remain') else module.in_channels
                out_channels = module.out_channels_remain if hasattr(module, 'out_channels_remain') else module.out_channels
                groups = module.groups_remain if hasattr(module, 'groups_remain') else module.groups
                n = PQ.get_percision(module.quantize_w, cfg).item()
                if not init_params:
                    parameters += in_channels // groups * out_channels * module.kernel_size ** 2 * n
                else:
                    parameters += in_channels // groups * out_channels * module.kernel_size ** 2 * 32

                if module.bias is not None:
                    if not init_params:
                        parameters += out_channels * n
                    else:
                        parameters += out_channels * 32
            elif isinstance(module, nn.BatchNorm2d):
                if module.affine:
                    num_features = module.num_features_remain if hasattr(module, 'num_features_remain') else module.num_features
                    parameters += num_features * 2 * 32
    return parameters


def add_feature_dimension(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_dimension_add = start_dimension_add.__get__(net_main_module)
    net_main_module.stop_dimension_add = stop_dimension_add.__get__(net_main_module)

    return net_main_module


def start_dimension_add(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    self.apply(add_feat_dim_hook_function)


def stop_dimension_add(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    self.apply(remove_feat_dim_hook_function)


def add_feat_dim_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, conv_dhp)):
            handle = module.register_forward_hook(conv_feat_dim_hook)
        elif isinstance(module, (nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6)):
            handle = module.register_forward_hook(relu_feat_dim_hook)
        elif isinstance(module, nn.Linear):
            handle = module.register_forward_hook(linear_feat_dim_hook)
        elif isinstance(module, nn.BatchNorm2d):
            handle = module.register_forward_hook(bn_feat_dim_hook)
        else:
            raise NotImplementedError('FLOPs calculation is not implemented for class {}'.format(module.__class__.__name__))
        module.__flops_handle__ = handle


def remove_feat_dim_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__


# ---- Internal functions
def is_supported_instance(module):
    if isinstance(module,
                  (
                          conv_dhp,
                          nn.Conv2d, nn.ConvTranspose2d,
                          nn.BatchNorm2d,
                          nn.Linear,
                          # nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6,
                  )):
        if hasattr(module, '__exclude_complexity__'):
            return False
        else:
            return True

    return False


def conv_feat_dim_hook(module, input, output):
    module.__output_dims__ = output.shape[2:]



def conv_calc_flops(self, nn):
    # Do not count bias addition
    batch_size = 1
    output_dims = np.prod(self.__output_dims__)

    kernel_dims = np.prod(self.kernel_size) if isinstance(self.kernel_size, tuple) else self.kernel_size ** 2
    in_channels = self.in_channels_remain if hasattr(self, 'in_channels_remain') else self.in_channels
    out_channels = self.out_channels_remain if hasattr(self, 'out_channels_remain') else self.out_channels
    groups = self.groups_remain if hasattr(self, 'groups_remain') else self.groups
    # groups = self.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_dims * in_channels * filters_per_channel

    active_elements_count = batch_size * output_dims

    overall_conv_flops = conv_per_position_flops * active_elements_count
    return int(overall_conv_flops)*nn


def relu_feat_dim_hook(module, input, output):
    s = output.shape
    module.__output_dims__ = s[2:]
    module.__output_channel__ = s[1]


def relu_calc_flops(self):
    batch = 1
    # TODO: relu channels attr is deleted
    channels = self.channels if hasattr(self, 'channels') else self.__output_channel__
    active_elements_count = batch * np.prod(self.__output_dims__) * channels
    # print(active_elements_count, id(self))
    # print(self)
    return int(active_elements_count)


def linear_feat_dim_hook(module, input, output):
    if len(output.shape[2:]) == 2:
        module.__additional_dims__ = 1
    else:
        module.__additional_dims__ = output.shape[1:-1]


def linear_calc_flops(self, nn):
    # Do not count bias addition
    batch_size = 1
    in_features = self.in_features_remain if hasattr(self, 'in_features_remain') else self.in_features
    out_features = self.out_features_remain if hasattr(self, 'out_features_remain') else self.out_features
    linear_flops = batch_size * np.prod(self.__additional_dims__) * in_features * out_features
    # print(self.in_features, in_features)
    return int(linear_flops)*nn


def bn_feat_dim_hook(module, input, output):
    module.__output_dims__ = output.shape[2:]


def bn_calc_flops(self):
    # Do not count bias addition
    batch = 1
    output_dims = np.prod(self.__output_dims__)
    channels = self.num_features_remain if hasattr(self, 'num_features_remain') else self.num_features
    batch_flops = batch * channels * output_dims
    # print(self.num_features, channels)
    if self.affine:
        batch_flops *= 2
    return int(batch_flops)

