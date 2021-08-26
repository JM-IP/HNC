"""
This is the base module of DHP: differentiable hyper pruning.
Define some default functions here.
"""

import torch.nn as nn
import torch
import os
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from model_dhp.quant_dorefa import *
import model_dhp.parametric_quantization as PQ
from util import utility
torch.backends.cudnn.deterministic = True
from torch.autograd import Variable


def plot_figure_dhp(latent_vector, l, filename):
    axis = np.array(list(range(1, latent_vector.shape[0]+1)))
    latent_vector = latent_vector.abs().detach().cpu().numpy()
    fig = plt.figure()
    plt.title('Latent vector {}, Max: {:.4f}, Ave: {:.4f}, Min: {:.4f}'.
              format(l, latent_vector.max(), latent_vector.mean(), latent_vector.min()))
    plt.plot(axis, latent_vector, label='Unsorted')
    plt.plot(axis, np.sort(latent_vector), label='Sorted')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Latent Vector')
    plt.grid(True)
    plt.savefig(filename, dpi=50)
    plt.close(fig)


def plot_per_layer_compression_ratio(per_layer_ratio, filename):
    axis = np.array(list(range(1, len(per_layer_ratio) + 1)))
    per_layer_ratio = np.array(per_layer_ratio)
    fig = plt.figure()
    plt.title('Per Layer Compression Ratio')
    plt.plot(axis, per_layer_ratio)
    plt.xlabel('Index')
    plt.ylabel('Compression Ratio')
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    # plt.show()
    plt.close(fig)


def plot_compression_ratio(compression_ratio, filename, frequency_per_epoch=1):
    if frequency_per_epoch == 1:
        axis = np.array(list(range(1, len(compression_ratio) + 1)))
    else:
        axis = np.array(list(range(1, len(compression_ratio) + 1)), dtype=float) / frequency_per_epoch
    compression_ratio = np.array(compression_ratio)
    fig = plt.figure()
    plt.title('Network Compression Ratio')
    plt.plot(axis, compression_ratio)
    # plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.grid(True)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def set_finetune_flag(module):
    module.finetuning = True
    if True:
        print('YJM is here!')
        for name, value in module.named_parameters():
            if name == 'd' or name == 'xmax' or name == 'n':
                value.trainbale = False



class conv_dhp_prune(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bias=False, groups=1, prune_groups=True, batchnorm=True,
                 act=True, location='after', concat=False, embedding_dim=8,
                 latent_vector=None,
                 std=1.0, cfg=None, args=None):
        #TODO: act should be set properly
        #TODO: constrain should be done
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param bias: whether bias is used in the convolutional layer
        :param groups: number of groups for group convolution or depth-wise convolution
        :param batchnorm: whether to append Batchnorm after the activations.
        :param act: whether to append ReLU after the activations.
        :param location: the location to put the batchnorm and act, choices 1) before -- before the convolution;
                                                                            2) after -- after the convolution.
        :param concat: whether to concatenate the input and output of the convolutional layer.
                        This is used for convolutions in DenseBlock.
        :param latent_vector: sometimes the latent vector is shared with another layer.
        :param embedding_dim: the dimension of the embedding space.
        """
        super(conv_dhp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.bias_flag = bias
        self.groups = groups
        self.prune_groups = prune_groups
        self.batchnorm = batchnorm
        self.act = act
        self.location = location

        self.concat = concat
        self.embedding_dim = embedding_dim
        self.finetuning = False
        self.in_channels_per_group = self.in_channels // self.groups
        self.cfg = cfg
        self.args = args
        self.regularization = args.regularization_factor
        # self.abit = abit
        # self.wbit = wbit
        # hypernetwork
        bound1 = math.sqrt(3) * math.sqrt(1 / 1)
        weight1 = torch.randn((self.in_channels_per_group, out_channels, embedding_dim)).uniform_(-bound1, bound1)
        # Hyperfan-in
        bound2 = math.sqrt(3) * math.sqrt(1 / (embedding_dim * self.in_channels_per_group * kernel_size ** 2))
        weight2 = torch.randn((self.in_channels_per_group, out_channels, kernel_size ** 2, embedding_dim)).uniform_(-bound2, bound2)

        if latent_vector is None:
            if self.groups == 1:
                self.latent_vector = nn.Parameter(torch.randn((out_channels)) * std)
            else:
                self.latent_vector = nn.Parameter(torch.randn((self.in_channels_per_group)) * std)
        else:
            self.latent_vector = latent_vector

        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias0 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels))
        self.bias1 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels, embedding_dim))
        self.bias2 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels, kernel_size ** 2))

        # main network
        self.bias = nn.Parameter(torch.zeros(out_channels)) if self.bias_flag else None
        if self.batchnorm:
            if self.location == 'before':
                # Used for DenseBlock, BatchNorm and ReLU appears before conv.
                self.bn_main = nn.BatchNorm2d(in_channels)
            else:
                # Used for other networks, BatchNorm and ReLU appears after conv.
                self.bn_main = nn.BatchNorm2d(out_channels)
        if bias:
            self.quantize_w, self.quantize_b = PQ.get_quantizers(w=None, cfg=self.cfg, delta_init=None, fix_parameters=True)
        else:
            self.quantize_w, _ = PQ.get_quantizers(w=None, cfg=self.cfg, delta_init=None, fix_parameters=True)
        if self.act:
            self.a_quant = PQ.activation_Q_fn(cfg)(fix_parameters=True)
        if self.args.cal_channel=='sign':
            self.calnumlayer=PQ.SignNoGradientfunc()
        elif self.args.cal_channel=='sigmoid':
            self.calnumlayer=torch.nn.Sigmoid()
        else:
            raise()
        # self.predefine = 2
    def __repr__(self):
        conv_str = 'Conv_dhp({}, {}, kernel_size=({}, {}), stride={}, bias={}, groups={}, concat={})'
        if hasattr(self, 'in_channels_remain') and hasattr(self, 'out_channels_remain') and hasattr(self, 'groups_remain'):
            conv_str = conv_str.format(self.in_channels_remain, self.out_channels_remain, self.kernel_size, self.kernel_size,
                                       self.stride, self.bias_flag, self.groups_remain, self.concat)
        else:
            conv_str = conv_str.format(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size,
                                       self.stride, self.bias_flag, self.groups, self.concat)
        s = ''
        if self.location == 'before':
            # DenseBlock
            if self.batchnorm:
                s = s + '\n(0): ' + repr(self.bn_main)
            if self.act:
                s = s + '\n(1): ' + repr(self.a_quant)
            s = s + '\n(2): ' + conv_str
        else:
            # the other convolutional layers
            s = s + '\n(0): ' + conv_str
            if self.batchnorm:
                s = s + '\n(1): ' + repr(self.bn_main)
            if self.act:
                s = s + '\n(2): ' + repr(self.a_quant)
        s = addindent(s, 2)
        return s

    def set_parameters(self, vector_mask, calc_weight=False):
        # latent_vector_input, mask_input, mask_output = vector_mask
        # t1,t2,t3,t4 = utility.timer(), utility.timer(), utility.timer(), utility.timer()
        # t1.tic()
        if self.groups == 1:
            # standard convolution
            latent_vector_input, mask_input, mask_output = vector_mask
            # TODO: check the mask_input and mask_output here
            latent_vector_output = self.latent_vector
            # self.latent_vector_real.detach().sign() * \
            #                    torch.max(torch.abs(self.latent_vector_real) - self.latent_vector_threshold,
            #                    Variable(torch.Tensor([0]), requires_grad=False).cuda())
            # added for mobilenetv3, repeat output channel for standard convolution, modified for RegNet.
            channel_repeat = self.out_channels // latent_vector_output.shape[0]
            if channel_repeat != 1:
                # print('channel_repeat {}'.format(channel_repeat))
                latent_vector_output = latent_vector_output.repeat_interleave(channel_repeat)
                mask_output = mask_output.repeat_interleave(channel_repeat)

        else:
            # depth-wise or group convolution, the latent input vector is fixed. The latent output vector is controlled
            # by the former layer
            latent_vector_input = self.latent_vector
            # self.latent_vector_real.detach().sign() * torch.max(torch.abs(self.latent_vector_real) -
            # self.latent_vector_threshold, Variable(torch.Tensor([0]), requires_grad=False).cuda())
            latent_vector_output, mask_output, mask_input = vector_mask
        # t1.hold()
        # t2.tic()
        mask_input = mask_input.to(torch.float32).nonzero().squeeze(1)
        mask_output = mask_output.to(torch.float32).nonzero().squeeze(1)
        # used to select the mask for the Batch Normalization layer
        mask_bn = mask_input if self.location == 'before' else mask_output
        # DEBUG INFORMATION
        # self.mask_input = mask_input
        # self.mask_output = mask_output
        # self.latent_vector_output = latent_vector_output
        # self.latent_vector_input = latent_vector_input
        # t2.hold()
        # t3.tic()
        # t1.tic()
        tmpmin = torch.min(latent_vector_input[mask_input].abs())
        if tmpmin>self.args.prune_threshold:
            tmpmin = self.args.prune_threshold

        # t1.hold()
        # t2.tic()
        self.in_channels_num = torch.sum(((self.calnumlayer(latent_vector_input.abs() -
                                                            tmpmin)+1)/2))
        # t2.hold()
        tmpmin = torch.min(latent_vector_output[mask_output].abs())
        if tmpmin>self.args.prune_threshold:
            tmpmin = self.args.prune_threshold
        self.out_channels_num = torch.sum(((self.calnumlayer(latent_vector_output.abs() -
                                                             tmpmin)+1)/2))
        # t3.hold()
        # t4.tic()
        if calc_weight:
            latent_vector_input = torch.index_select(latent_vector_input, dim=0, index=mask_input)
            latent_vector_output = torch.index_select(latent_vector_output, dim=0, index=mask_output)
            weight = self.calc_weight(latent_vector_input, latent_vector_output, mask_input, mask_output)

            bias = self.bias if self.bias is None else torch.index_select(self.bias, dim=0, index=mask_output)
            self.weight = nn.Parameter(weight)
            self.bias = bias if bias is None else nn.Parameter(bias)

            if self.batchnorm:
                bn_weight = torch.index_select(self.bn_main.weight, dim=0, index=mask_bn)
                bn_bias = torch.index_select(self.bn_main.bias, dim=0, index=mask_bn)
                bn_mean = torch.index_select(self.bn_main.running_mean, dim=0, index=mask_bn)
                bn_var = torch.index_select(self.bn_main.running_var, dim=0, index=mask_bn)

                self.bn_main.weight = nn.Parameter(bn_weight)
                self.bn_main.bias = nn.Parameter(bn_bias)
                self.bn_main.running_mean = bn_mean
                self.bn_main.running_var = bn_var

        # self.in_channels_remain = mask_input.shape[0]
        if self.groups == 1:
            self.in_channels_remain = mask_input.shape[0]
            self.groups_remain = self.groups
        else:
            self.in_channels_remain = mask_output.shape[0]
            if self.prune_groups:
                self.groups_remain = mask_output.shape[0] // self.in_channels_per_group # modified for group convolution
            else:
                self.groups_remain = self.groups
            if calc_weight:
                if self.prune_groups:
                    self.groups = mask_output.shape[0] // self.in_channels_per_group
        self.out_channels_remain = mask_output.shape[0]


        if self.batchnorm:
            self.bn_main.num_features_remain = mask_bn.shape[0]
            self.bn_main.num_features = mask_bn.shape[0]
        # t4.hold()
        # print("yjm is here to debug time setp is:", t1.release(),t2.release(),t3.release(),t4.release())
    def calc_weight(self, latent_vector_input=None, latent_vector_output=None, mask_input=None, mask_output=None):
        """
        This function is used by self.set_parameters and self.forward.
        1) When used by self.forward, the mask_input and mask_output are not provided. Thus,
           this function only calculates the weight parameter from the hyper-network for
           the pruning stage.
        2) When used by self.set_parameters, mask_input and mask_output are provided. Because they are used to set the
            remaining number of input and output channels and to calculate the pruned weight parameter. In this mode,
            the latent_vector_input and latent_vector_output are also provided. (All of the four arguments are provided.)

        :param latent_vector_input:
        :param latent_vector_output:
        :param mask_input:
        :param mask_output:
        :return:
        """

        # deal with the case that the function is used by self.forward
        # when latent_vector_output is not None, it is provided by the former layer. This is True for depth-wise conv.
        # when latent_vector_output is None, it is stored in the current layer. This is the normal case
        # print("DEBUG: model is :", self)
        if latent_vector_output is None:
            latent_vector_output = self.latent_vector
            # print("DEBUG: latent_vector_input is not None and is", latent_vector_input)
            # print("DEBUG: latent_vector_output is None and set as", latent_vector_output)

            # print("DEBUG: latent_vector_real is None and set as", self.latent_vector_real)
            # self.latent_vector_real.detach().sign() * torch.max(torch.abs(self.latent_vector_real) -
            # self.latent_vector_threshold, Variable(torch.Tensor([0]), requires_grad=False).cuda())

            # # added for mobilenetv3, modified for RegNet (group convolution)

            channel_repeat = self.out_channels // latent_vector_output.shape[0]
            if channel_repeat != 1:
                latent_vector_output = latent_vector_output.repeat_interleave(channel_repeat)
                # if mask_output is not None:
                #     mask_output = mask_output.repeat_interleave(4)

        # when latent_vector_input is not None, it is provided by the former layers. This is the normal case. And it controls the input dimension of the current conv.
        # When latent_vector_input is None, it is stored in the current layer. This is the case for depth-wise conv.
        elif latent_vector_input is None:
            latent_vector_input = self.latent_vector
            # print("DEBUG: latent_vector_input is None and set as", latent_vector_input)
            # print("DEBUG: latent_vector_output is not None and is", latent_vector_output)

            # self.latent_vector_real.detach().sign() * torch.max(torch.abs(self.latent_vector_real) -
            # self.latent_vector_threshold, Variable(torch.Tensor([0]), requires_grad=False).cuda())

        # at least one from latent_vector_input and latent_vector_output should be provided
        elif latent_vector_output is None and latent_vector_input is None:
            raise NotImplementedError('During the pruning of MobileNetV2 convs, please provide at least one of the latent input and output vectors')
        # The last case is that both latent_vector_input, latent_vector_output are provided, this is the case for self.set_parameters.

        if mask_input is None and mask_output is None:
            # self.forward()
            bias0, bias1, bias2, weight1, weight2 = self.bias0, self.bias1, self.bias2, self.weight1, self.weight2
        else:
            bias0 = torch.index_select(torch.index_select(self.bias0, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            bias1 = torch.index_select(torch.index_select(self.bias1, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            bias2 = torch.index_select(torch.index_select(self.bias2, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            weight1 = torch.index_select(torch.index_select(self.weight1, dim=0, index=mask_input), dim=1,
                                         index=mask_output)
            weight2 = torch.index_select(torch.index_select(self.weight2, dim=0, index=mask_input), dim=1,
                                         index=mask_output)

        weight = torch.matmul(latent_vector_input.unsqueeze(-1), latent_vector_output.unsqueeze(0)) + bias0
        weight = weight.unsqueeze(-1) * weight1 + bias1
        weight = torch.matmul(weight2, weight.unsqueeze(-1)).squeeze(-1) + bias2
        # if weight.nelement() != self.in_channels * self.out_channels * self.kernel_size ** 2:
        #     embed()
        in_channels = latent_vector_input.nelement()
        out_channels = latent_vector_output.nelement()
        weight = weight.reshape(in_channels, out_channels, self.kernel_size, self.kernel_size).permute(1, 0, 2, 3)
        # weight = self.bn_hyper(weight)

        return weight

    def forward(self, x, getq=False):
        timer_getweight, timer_forward = utility.timer(), utility.timer()
        timeall=[]
        # calculate the weights from the hypernetwork
        timer_getweight.tic()
        if not self.finetuning:
            if self.groups == 1:
                x, latent_vector_input = x
                weight = self.calc_weight(latent_vector_input=latent_vector_input)
            else:
                x, latent_vector_output = x
                weight = self.calc_weight(latent_vector_output=latent_vector_output)
        timer_getweight.hold()
        # if getq:
        #     # print("=================>>yjm is here, get quantizer", self.predefine)
        #
        #     if self.bias_flag:
        #         self.quantize_w, self.quantize_b = PQ.get_quantizers(w=weight, cfg=self.cfg)
        #         self.quantize_b = self.quantize_b.cuda()
        #     else:
        #         self.quantize_w, _ = PQ.get_quantizers(w=weight, cfg=self.cfg)
        #     self.quantize_w = self.quantize_w.cuda()

        # # print("yjm is here!!!", self.cfg.w_bitwidth)
        # # when the batch normalization appears before the convolution
        timer_forward.tic()
        if self.location == 'before':
            # Note that in the Dense Block and Transition Block of DenseNet,
            # the Batch Normalization layer and ReLU layer
            # always appear before the convolution. But in Dense Block,
            # concatenation is used while in Transition Block,
            # concatenation is not used.
            if self.batchnorm and self.act:
                out = self.a_quant(self.bn_main(x))
            else:
                raise NotImplementedError('When the Batch Normalization layer and '
                                          'a_quant layer appears before the convolution,'
                                          'they must appear or disappear together')
        else:
            out = x
        timer_forward.hold()
        timeall.append(timer_forward.release())
        timer_forward.tic()
        # apply the convolution
        if not self.finetuning:
            # out = self.a_quant(out)
            if self.quantize_w is not None:
                weight = self.quantize_w(weight)
            bias = self.bias
            if self.bias is not None and self.quantize_b is not None:
                bias = self.quantize_b(self.bias)
            timer_forward.hold()
            timeall.append(timer_forward.release())
            timer_forward.tic()
            out = F.conv2d(out, weight, bias=bias, stride=self.stride, padding=self.kernel_size // 2, groups=self.groups).cuda()
        else:
            # out = self.a_quant(out)
            weight = self.quantize_w(self.weight)
            bias = self.bias
            if self.bias is not None and self.quantize_b is not None:
                bias = self.quantize_b(self.bias)
            out = F.conv2d(out, weight, bias=bias, stride=self.stride, padding=self.kernel_size // 2, groups=self.groups)

        # when the batch normalization appears after the convolution
        if self.location == 'before':
            # Dense Block use concatenation
            if self.concat:
                out = torch.cat((x, out), 1)
        else:
            if self.batchnorm:
                out = self.bn_main(out)
            if self.act:
                out = self.a_quant(out)
        # if self.predefine>0:
        #     print("=================>>yjm is here done", self.predefine)
        #     self.predefine = self.predefine-1
        #     print("=================>>yjm is here after", self.predefine)
        timer_forward.hold()
        timeall.append(timer_forward.release())
        # print('yjm is here to debug forward:', timer_getweight.release(), timeall)
        return out

class conv_dhp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bias=False, groups=1, prune_groups=True, batchnorm=True,
                 act=True, location='after', concat=False, embedding_dim=8,
                 latent_vector=None,
                 std=1.0, cfg=None, args=None, init_weight=None, quant=True):
        #TODO: act should be set properly
        #TODO: constrain should be done
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param bias: whether bias is used in the convolutional layer
        :param groups: number of groups for group convolution or depth-wise convolution
        :param batchnorm: whether to append Batchnorm after the activations.
        :param act: whether to append ReLU after the activations.
        :param location: the location to put the batchnorm and act, choices 1) before -- before the convolution;
                                                                            2) after -- after the convolution.
        :param concat: whether to concatenate the input and output of the convolutional layer.
                        This is used for convolutions in DenseBlock.
        :param latent_vector: sometimes the latent vector is shared with another layer.
        :param embedding_dim: the dimension of the embedding space.
        """
        super(conv_dhp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.bias_flag = bias
        self.groups = groups
        self.prune_groups = prune_groups
        self.batchnorm = batchnorm
        self.act = act
        self.location = location

        self.concat = concat
        self.embedding_dim = embedding_dim
        self.finetuning = False
        self.in_channels_per_group = self.in_channels // self.groups
        self.cfg = cfg
        self.args = args
        self.regularization = args.regularization_factor

        if False and init_weight:
            weight1 = torch.ones((self.in_channels_per_group, out_channels, embedding_dim))
            weight2 = torch.ones((self.in_channels_per_group, out_channels, kernel_size ** 2, embedding_dim))
            if latent_vector is None:
                if self.groups == 1:
                    self.latent_vector = nn.Parameter(torch.ones((out_channels)))
                else:
                    self.latent_vector = nn.Parameter(torch.ones((self.in_channels_per_group)))
            else:
                self.latent_vector = latent_vector
            # hypernetwork
        else:
            bound1 = math.sqrt(3) * math.sqrt(1 / 1)
            weight1 = torch.randn((self.in_channels_per_group, out_channels, embedding_dim)).uniform_(-bound1, bound1)
            # Hyperfan-in
            bound2 = math.sqrt(3) * math.sqrt(1 / (embedding_dim * self.in_channels_per_group * kernel_size ** 2))
            weight2 = torch.randn((self.in_channels_per_group, out_channels, kernel_size ** 2, embedding_dim)).uniform_(-bound2, bound2)

            if latent_vector is None:
                if self.groups == 1:
                    self.latent_vector = nn.Parameter(torch.randn((out_channels)) * std)
                else:
                    self.latent_vector = nn.Parameter(torch.randn((self.in_channels_per_group)) * std)
            else:
                self.latent_vector = latent_vector

        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias0 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels))
        self.bias1 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels, embedding_dim))

        self.bias2 = nn.Parameter(torch.zeros(self.in_channels_per_group, out_channels, kernel_size ** 2))

        # main network
        self.bias = nn.Parameter(torch.zeros(out_channels)) if self.bias_flag else None
        if self.batchnorm:
            if self.location == 'before':
                # Used for DenseBlock, BatchNorm and ReLU appears before conv.
                self.bn_main = nn.BatchNorm2d(in_channels)
            else:
                # Used for other networks, BatchNorm and ReLU appears after conv.
                self.bn_main = nn.BatchNorm2d(out_channels)
        if quant:
            if bias:
                self.quantize_w, self.quantize_b = PQ.get_quantizers(w=None, cfg=self.cfg, delta_init=cfg.w_stepsize)
            else:
                self.quantize_w, _ = PQ.get_quantizers(w=None, cfg=self.cfg, delta_init=cfg.w_stepsize)
        else:
            self.quantize_w = None
            self.quantize_b = None
        self.init_q = True
        if self.act:
            self.a_quant = PQ.activation_Q_fn(cfg)()
        if self.args.cal_channel=='sign':
            self.calnumlayer=PQ.SignNoGradientfunc()
        elif self.args.cal_channel=='sigmoid':
            self.calnumlayer=torch.nn.Sigmoid()
        else:
            raise()
        # self.predefine = 2
    def __repr__(self):
        conv_str = 'Conv_dhp({}, {}, kernel_size=({}, {}), stride={}, bias={}, groups={}, concat={})'
        if hasattr(self, 'in_channels_remain') and hasattr(self, 'out_channels_remain') and hasattr(self, 'groups_remain'):
            conv_str = conv_str.format(self.in_channels_remain, self.out_channels_remain, self.kernel_size, self.kernel_size,
                                       self.stride, self.bias_flag, self.groups_remain, self.concat)
        else:
            conv_str = conv_str.format(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size,
                                       self.stride, self.bias_flag, self.groups, self.concat)
        s = ''
        if self.location == 'before':
            # DenseBlock
            if self.batchnorm:
                s = s + '\n(0): ' + repr(self.bn_main)
            if self.act:
                s = s + '\n(1): ' + repr(self.a_quant)
            s = s + '\n(2): ' + conv_str
        else:
            # the other convolutional layers
            s = s + '\n(0): ' + conv_str
            if self.batchnorm:
                s = s + '\n(1): ' + repr(self.bn_main)
            if self.act:
                s = s + '\n(2): ' + repr(self.a_quant)
        s = addindent(s, 2)
        return s

    def set_parameters(self, vector_mask, calc_weight=False):
        # latent_vector_input, mask_input, mask_output = vector_mask
        # t1,t2,t3,t4 = utility.timer(), utility.timer(), utility.timer(), utility.timer()
        # t1.tic()
        if self.groups == 1:
            # standard convolution
            latent_vector_input, mask_input, mask_output = vector_mask
            # TODO: check the mask_input and mask_output here
            latent_vector_output = self.latent_vector
            # self.latent_vector_real.detach().sign() * \
            #                    torch.max(torch.abs(self.latent_vector_real) - self.latent_vector_threshold,
            #                    Variable(torch.Tensor([0]), requires_grad=False).cuda())
            # added for mobilenetv3, repeat output channel for standard convolution, modified for RegNet.
            channel_repeat = self.out_channels // latent_vector_output.shape[0]
            if channel_repeat != 1:
                # print('channel_repeat {}'.format(channel_repeat))
                latent_vector_output = latent_vector_output.repeat_interleave(channel_repeat)
                mask_output = mask_output.repeat_interleave(channel_repeat)

        else:
            # depth-wise or group convolution, the latent input vector is fixed. The latent output vector is controlled
            # by the former layer
            latent_vector_input = self.latent_vector
            # self.latent_vector_real.detach().sign() * torch.max(torch.abs(self.latent_vector_real) -
            # self.latent_vector_threshold, Variable(torch.Tensor([0]), requires_grad=False).cuda())
            latent_vector_output, mask_output, mask_input = vector_mask
        # t1.hold()
        # t2.tic()
        mask_input = mask_input.to(torch.float32).nonzero().squeeze(1)
        mask_output = mask_output.to(torch.float32).nonzero().squeeze(1)
        # used to select the mask for the Batch Normalization layer
        mask_bn = mask_input if self.location == 'before' else mask_output
        # DEBUG INFORMATION
        # self.mask_input = mask_input
        # self.mask_output = mask_output
        # self.latent_vector_output = latent_vector_output
        # self.latent_vector_input = latent_vector_input
        # t2.hold()
        # t3.tic()
        # t1.tic()
        tmpmin = torch.min(latent_vector_input[mask_input].abs())
        if tmpmin>self.args.prune_threshold:
            tmpmin = self.args.prune_threshold

        # t1.hold()
        # t2.tic()
        self.in_channels_num = torch.sum(((self.calnumlayer(latent_vector_input.abs() -
                                                            tmpmin)+1)/2))
        # t2.hold()
        tmpmin = torch.min(latent_vector_output[mask_output].abs())
        if tmpmin>self.args.prune_threshold:
            tmpmin = self.args.prune_threshold
        self.out_channels_num = torch.sum(((self.calnumlayer(latent_vector_output.abs() -
                                                             tmpmin)+1)/2))
        # t3.hold()
        # t4.tic()
        if calc_weight:
            latent_vector_input = torch.index_select(latent_vector_input, dim=0, index=mask_input)
            latent_vector_output = torch.index_select(latent_vector_output, dim=0, index=mask_output)
            weight = self.calc_weight(latent_vector_input, latent_vector_output, mask_input, mask_output)

            bias = self.bias if self.bias is None else torch.index_select(self.bias, dim=0, index=mask_output)
            self.weight = nn.Parameter(weight)
            self.bias = bias if bias is None else nn.Parameter(bias)

            if self.batchnorm:
                bn_weight = torch.index_select(self.bn_main.weight, dim=0, index=mask_bn)
                bn_bias = torch.index_select(self.bn_main.bias, dim=0, index=mask_bn)
                bn_mean = torch.index_select(self.bn_main.running_mean, dim=0, index=mask_bn)
                bn_var = torch.index_select(self.bn_main.running_var, dim=0, index=mask_bn)

                self.bn_main.weight = nn.Parameter(bn_weight)
                self.bn_main.bias = nn.Parameter(bn_bias)
                self.bn_main.running_mean = bn_mean
                self.bn_main.running_var = bn_var

        # self.in_channels_remain = mask_input.shape[0]
        if self.groups == 1:
            self.in_channels_remain = mask_input.shape[0]
            self.groups_remain = self.groups
        else:
            self.in_channels_remain = mask_output.shape[0]
            if self.prune_groups:
                self.groups_remain = mask_output.shape[0] // self.in_channels_per_group # modified for group convolution
            else:
                self.groups_remain = self.groups
            if calc_weight:
                if self.prune_groups:
                    self.groups = mask_output.shape[0] // self.in_channels_per_group
        self.out_channels_remain = mask_output.shape[0]


        if self.batchnorm:
            self.bn_main.num_features_remain = mask_bn.shape[0]
            self.bn_main.num_features = mask_bn.shape[0]
        # t4.hold()
        # print("yjm is here to debug time setp is:", t1.release(),t2.release(),t3.release(),t4.release())
    def calc_weight(self, latent_vector_input=None, latent_vector_output=None, mask_input=None, mask_output=None):
        """
        This function is used by self.set_parameters and self.forward.
        1) When used by self.forward, the mask_input and mask_output are not provided. Thus,
           this function only calculates the weight parameter from the hyper-network for
           the pruning stage.
        2) When used by self.set_parameters, mask_input and mask_output are provided. Because they are used to set the
            remaining number of input and output channels and to calculate the pruned weight parameter. In this mode,
            the latent_vector_input and latent_vector_output are also provided. (All of the four arguments are provided.)

        :param latent_vector_input:
        :param latent_vector_output:
        :param mask_input:
        :param mask_output:
        :return:
        """

        # deal with the case that the function is used by self.forward
        # when latent_vector_output is not None, it is provided by the former layer. This is True for depth-wise conv.
        # when latent_vector_output is None, it is stored in the current layer. This is the normal case
        # print("DEBUG: model is :", self)
        if latent_vector_output is None:
            latent_vector_output = self.latent_vector
            # print("DEBUG: latent_vector_input is not None and is", latent_vector_input)
            # print("DEBUG: latent_vector_output is None and set as", latent_vector_output)

            # print("DEBUG: latent_vector_real is None and set as", self.latent_vector_real)
            # self.latent_vector_real.detach().sign() * torch.max(torch.abs(self.latent_vector_real) -
            # self.latent_vector_threshold, Variable(torch.Tensor([0]), requires_grad=False).cuda())

            # # added for mobilenetv3, modified for RegNet (group convolution)

            channel_repeat = self.out_channels // latent_vector_output.shape[0]
            if channel_repeat != 1:
                latent_vector_output = latent_vector_output.repeat_interleave(channel_repeat)
                # if mask_output is not None:
                #     mask_output = mask_output.repeat_interleave(4)

        # when latent_vector_input is not None, it is provided by the former layers. This is the normal case. And it controls the input dimension of the current conv.
        # When latent_vector_input is None, it is stored in the current layer. This is the case for depth-wise conv.
        elif latent_vector_input is None:
            latent_vector_input = self.latent_vector
            # print("DEBUG: latent_vector_input is None and set as", latent_vector_input)
            # print("DEBUG: latent_vector_output is not None and is", latent_vector_output)

            # self.latent_vector_real.detach().sign() * torch.max(torch.abs(self.latent_vector_real) -
            # self.latent_vector_threshold, Variable(torch.Tensor([0]), requires_grad=False).cuda())

        # at least one from latent_vector_input and latent_vector_output should be provided
        elif latent_vector_output is None and latent_vector_input is None:
            raise NotImplementedError('During the pruning of MobileNetV2 convs, please provide at least one of the latent input and output vectors')
        # The last case is that both latent_vector_input, latent_vector_output are provided, this is the case for self.set_parameters.

        if mask_input is None and mask_output is None:
            # self.forward()
            bias0, bias1, bias2, weight1, weight2 = self.bias0, self.bias1, self.bias2, self.weight1, self.weight2
        else:
            bias0 = torch.index_select(torch.index_select(self.bias0, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            bias1 = torch.index_select(torch.index_select(self.bias1, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            bias2 = torch.index_select(torch.index_select(self.bias2, dim=0, index=mask_input), dim=1,
                                       index=mask_output)
            weight1 = torch.index_select(torch.index_select(self.weight1, dim=0, index=mask_input), dim=1,
                                         index=mask_output)
            weight2 = torch.index_select(torch.index_select(self.weight2, dim=0, index=mask_input), dim=1,
                                         index=mask_output)

        weight = torch.matmul(latent_vector_input.unsqueeze(-1), latent_vector_output.unsqueeze(0)) + bias0
        weight = weight.unsqueeze(-1) * weight1 + bias1
        weight = torch.matmul(weight2, weight.unsqueeze(-1)).squeeze(-1) + bias2
        # if weight.nelement() != self.in_channels * self.out_channels * self.kernel_size ** 2:
        #     embed()
        in_channels = latent_vector_input.nelement()
        out_channels = latent_vector_output.nelement()
        weight = weight.reshape(in_channels, out_channels, self.kernel_size, self.kernel_size).permute(1, 0, 2, 3)
        # weight = self.bn_hyper(weight)
        if not self.init_q:
            if self.bias_flag:
                quantize_w, quantize_b = PQ.get_quantizers(w=weight, cfg=self.cfg)
                self.quantize_b.d.data = quantize_b.d.data.cuda()
                self.quantize_b.xmax.data = quantize_b.xmax.data.cuda()
            else:
                quantize_w, _ = PQ.get_quantizers(w=weight, cfg=self.cfg)
            self.quantize_w.d.data = quantize_w.d.data.cuda()
            self.quantize_w.xmax.data = quantize_w.xmax.data.cuda()

            self.init_q = True
        return weight

    def forward(self, x, getq=False):
        timer_getweight, timer_forward = utility.timer(), utility.timer()
        timeall=[]
        # calculate the weights from the hypernetwork
        timer_getweight.tic()
        if not self.finetuning:
            if self.groups == 1:
                x, latent_vector_input = x
                weight = self.calc_weight(latent_vector_input=latent_vector_input)
            else:
                x, latent_vector_output = x
                weight = self.calc_weight(latent_vector_output=latent_vector_output)
        timer_getweight.hold()
        # if getq:
        #     # print("=================>>yjm is here, get quantizer", self.predefine)
        #
        #     if self.bias_flag:
        #         self.quantize_w, self.quantize_b = PQ.get_quantizers(w=weight, cfg=self.cfg)
        #         self.quantize_b = self.quantize_b.cuda()
        #     else:
        #         self.quantize_w, _ = PQ.get_quantizers(w=weight, cfg=self.cfg)
        #     self.quantize_w = self.quantize_w.cuda()

        # # print("yjm is here!!!", self.cfg.w_bitwidth)
        # # when the batch normalization appears before the convolution
        timer_forward.tic()
        if self.location == 'before':
            # Note that in the Dense Block and Transition Block of DenseNet,
            # the Batch Normalization layer and ReLU layer
            # always appear before the convolution. But in Dense Block,
            # concatenation is used while in Transition Block,
            # concatenation is not used.
            if self.batchnorm and self.act:
                out = self.a_quant(self.bn_main(x))
            else:
                raise NotImplementedError('When the Batch Normalization layer and '
                                          'a_quant layer appears before the convolution,'
                                          'they must appear or disappear together')
        else:
            out = x
        timer_forward.hold()
        timeall.append(timer_forward.release())
        timer_forward.tic()
        # apply the convolution
        if not self.finetuning:
            # out = self.a_quant(out)
            if self.quantize_w is not None:
                weight = self.quantize_w(weight)
            bias = self.bias
            if self.bias is not None and self.quantize_b is not None:
                bias = self.quantize_b(self.bias)
            timer_forward.hold()
            timeall.append(timer_forward.release())
            timer_forward.tic()
            out = F.conv2d(out, weight, bias=bias, stride=self.stride, padding=self.kernel_size // 2, groups=self.groups).cuda()
        else:
            # out = self.a_quant(out)
            weight = self.weight
            if self.quantize_w is not None:
                weight = self.quantize_w(self.weight)
            bias = self.bias
            if self.bias is not None and self.quantize_b is not None:
                bias = self.quantize_b(self.bias)
            out = F.conv2d(out, weight, bias=bias, stride=self.stride, padding=self.kernel_size // 2, groups=self.groups)

        # when the batch normalization appears after the convolution
        if self.location == 'before':
            # Dense Block use concatenation
            if self.concat:
                out = torch.cat((x, out), 1)
        else:
            if self.batchnorm:
                out = self.bn_main(out)
            # print("output before quant",out)
            if self.act:
                out = self.a_quant(out)
            # print("output after quant",out)
        # if self.predefine>0:
        #     print("=================>>yjm is here done", self.predefine)
        #     self.predefine = self.predefine-1
        #     print("=================>>yjm is here after", self.predefine)
        timer_forward.hold()
        timeall.append(timer_forward.release())
        # print('yjm is here to debug forward:', timer_getweight.release(), timeall)
        return out

class DHP_Base(nn.Module):

    def __init__(self, args):
        super(DHP_Base, self).__init__()
        self.args = args
        self.kernel_size = args.kernel_size
        self.finetuning = False  # used to select the forward pass
        self.regularization = args.regularization_factor  # the regularization factor for l1 sparsity regularizer
        self.pt = args.prune_threshold  # The pruning threshold
        self.mc = args.mc  # The minimum number of remaining channels
        self.rp = args.remain_percentage
        self.embedding_dim = args.embedding_dim
        self.grad_prune = args.grad_prune
        self.grad_normalize = args.grad_normalize

        if args.data_train == 'ImageNet':
            self.n_classes = 1000
        elif args.data_train.find('CIFAR') >= 0:
            self.n_classes = int(args.data_train[5:])
        elif args.data_train == 'Tiny_ImageNet':
            self.n_classes = 200
        else:
            raise NotImplementedError('Dataset {} not implemented for pruing DenseNet'.format(args.data_train))

    def gather_latent_vector(self, return_key=False, grad_prune=False,
                             grad_normalize=False):
        """
        :return: all of the latent vectors
        """
        latent_vectors = []
        s = []
        kk = []
        for k, v in self.state_dict(keep_vars=True).items():
            if k.find('latent_vector') >= 0 and not id(v) in s:
                # print(k)
                assert not grad_prune
                if grad_prune and v.grad is not None:
                    vector = v.grad
                    if grad_normalize:
                        vector /= vector.max()
                else:
                    # vector = v.cuda()
                    vector = v

                latent_vectors.append(vector)
                s.append(id(v))
                if return_key:
                    kk.append(k)
        if return_key:
            return latent_vectors, kk
        else:
            return latent_vectors

    def show_latent_vector(self):
        latent_vectors, kk = self.gather_latent_vector(return_key=True)
        for i, (k, v) in enumerate(zip(kk, latent_vectors)):
            print(i, list(v.shape), k)

    def mask(self):
        """
        :return: Return the mask of all of the latent vectors used to set the remained number of channels and to
                calculate the weights from the hypernetwork after the proximal gradient optimization.
        """
        pass

    def soft_thresholding(self, latent_vector, reg):
        """
        This is the soft-thresholding function, namely the closed-form solution of proximal operator with l1
        regularization. The latent vector is updated by proximal gradient.
        :param latent_vector: the latent vector with l1 regularization.
        :param reg: the regularization factor.
        """
        vector = latent_vector.data
        vector = torch.max(torch.abs(vector) - reg, torch.zeros_like(vector, device=vector.device)) * torch.sign(vector)
        latent_vector.data = vector

    def proximal_euclidean_norm(self, latent_vector, reg):
        vector = latent_vector.data
        # reg = reg * vector.shape[0]
        scale = 1 - reg / torch.max(torch.norm(latent_vector), torch.tensor(reg, device=vector.device))
        vector = scale * vector
        latent_vector.data = vector

    def remain_channels(self, vector, percentage=None):
        if percentage is not None:
            channels = int(vector.shape[0] * percentage)
        else:
            if self.rp == -1:
                channels = self.mc
            else:
                channels = max(int(vector.shape[0] * self.rp), 1)
        return channels

    def proximal_operator(self, lr):
        """
        This function applies proximal operator to the latent vectors that need to be pruned, which could be different
        for every network.
        :param lr: the learning rate of SGD
        """
        pass

    def set_parameters(self, calc_weight=False):
        """
        Set the remained number of channels used to calculate the compression ratio. And calculate the weights from the
        hypernetwork if calc_weight is True. This is done at the end of pruning stage.
        """
        pass

    def forward(self, x):
        """
        The forward pass
        :param x: the input of the network
        :return: the output of the network
        """
        pass

    def delete_hyperparameters(self):
        """
        delete the hyperparameters including the latent vectors, weights and biases in the hypernetwork.
        :return:
        """
        for m in self.modules():
            if isinstance(m, conv_dhp):
                # del m.latent_vector, m.weight1, m.weight2, m.bias0, m.bias1, m.bias2
                del m.latent_vector, m.weight1, m.weight2, m.bias0, m.bias1, m.bias2

    def reset_after_searching(self):
        self.apply(set_finetune_flag)
        self.set_parameters(calc_weight=True)
        self.delete_hyperparameters()

    def set_channels(self):
        for m in self.modules():
            if isinstance(m, conv_dhp):
                s = m.weight.shape
                if m.groups == 1:
                    m.in_channels, m.out_channels = s[1], s[0]
                    m.in_channels_remain, m.out_channels_remain = s[1], s[0]
                else:
                    m.in_channels, m.out_channels = s[0], s[0]
                    m.in_channels_remain, m.out_channels_remain = s[0], s[0]
                    m.groups = s[0]
                    m.groups_remain = s[0]
                # print(s)
            elif isinstance(m, nn.BatchNorm2d):
                s = m.weight.shape
                m.num_features, m.num_features_remain = s[0], s[0]
            elif isinstance(m, nn.Linear):
                s = m.weight.shape
                m.in_features, m.out_features = s[1], s[0]
                m.in_features_remain, m.out_features_remain = s[1], s[0]

    def load_state_dict(self, state_dict, strict=True):
        """
        load state dictionary
        """
        if strict:
            # used to load the model parameters during training
            super(DHP_Base, self).load_state_dict(state_dict, strict)
        else:
            # used to load the model parameters during test
            own_state = self.state_dict(keep_vars=True)
            for name, param in state_dict.items():
                if name in own_state:
                    if isinstance(param, nn.Parameter):
                        param = param.data
                    if param.size() != own_state[name].size():
                        own_state[name].data = param
                    else:
                        own_state[name].data.copy_(param)
            self.set_channels()

    # def load(self, args, strict=True):
    #     """Not sure whether this is really used"""
    #     if args.pretrained:
    #         self.load_state_dict(torch.load(args.pretrained), strict=strict)

    def latent_vector_distribution(self, epoch, batch, fpath):
        """
        save the distribution of the latent vectors
        :param epoch:
        :param batch:
        :param fpath:
        :return:
        """
        filename = os.path.join(fpath, 'features/latent{}/epoch{}_batch{}.pdf')
        latent_vector_list=self.gather_latent_vector()
        for i, v in enumerate(latent_vector_list):
            if not os.path.exists(os.path.join(fpath, 'features/latent{}'.format(i + 1))):
                os.makedirs(os.path.join(fpath, 'features/latent{}'.format(i + 1)))
            plot_figure_dhp(latent_vector_list[i].data, i + 1, filename.format(i + 1, epoch, batch))

    def per_layer_compression_ratio_quantize_precision(self, epoch, batch, fpath, save_pt=False, cfg=None):
        """
        save layer-wise compression ratio.
        :param epoch:
        :param batch:
        :param fpath:
        :param save_pt:
        :return:
        """
        per_layer_ratio = []
        per_layer_precision = []
        per_layer_act_precision = []
        for name, m in self.named_modules():
            if isinstance(m, conv_dhp):
                per_layer_precision.append(PQ.get_percision(m.quantize_w, cfg).item())
                if 'downsample' in name:
                    per_layer_act_precision.append(-1)
            if '.act_out.a_quant' in name or '.a_quant.a_quant' in name:
                per_layer_act_precision.append(PQ.get_percision_a(m, cfg).item())

        layers = [m for m in self.modules() if isinstance(m, conv_dhp)]
        # compute layer-wise compression ratio in terms of remaining number of channels
        for l, layer in enumerate(layers):
            per_layer_ratio.append(layer.out_channels_remain / layer.out_channels)
        #
        #
        # for l in layers:
        #     per_layer_precision.append(PQ.get_percision(l.quantize_w, cfg).item())
        #     try:
        #         per_layer_act_precision.append(PQ.get_percision(l.a_quant.a_quant, cfg).item())
        #     except:
        #         per_layer_act_precision.append(-1)
        # the way to save the layer-wise compression ratio
        per_layer_act_precision.append(32)# for final layer, del if error
        if not save_pt:
            plot_per_layer_ratio_quantize_precision(per_layer_ratio, per_layer_precision, per_layer_act_precision,
                                                    os.path.join(fpath, 'per_layer_compression_ratio_quantize_precision/epoch{}_batch{}.pdf'.format(epoch, batch)))
        else:
            torch.save(per_layer_ratio, os.path.join(fpath, 'per_layer_compression_ratio_quantize_precision_final.pt'))


def plot_per_layer_ratio_quantize_precision(per_layer_ratio, per_layer_precision,
                                            per_layer_act_precision, filename):
    axis = np.array(list(range(1, len(per_layer_ratio) + 1)))
    per_layer_ratio = np.array(per_layer_ratio)
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    plt.grid(True)
    ax1 = ax2.twinx()

    ax1.set_ylabel('Compression Ratio')
    ax1.plot(axis, per_layer_ratio, 'r-', label='Prune ratio')
    print(len(per_layer_ratio), len(per_layer_precision))
    ax2.plot(axis, per_layer_precision, 'bx', label='Weight precision')
    ax2.set_ylabel('Percision')
    print(len(per_layer_act_precision))
    ax2.plot(axis, per_layer_act_precision, 'go', label='Activation precision')
    ax1.set_ylim(0, 1.05)
    # ax2.set_ylim(min())
    plt.title('Per Layer Compression Ratio')
    plt.xlabel('Index')
    plt.xticks(list(range(1, len(per_layer_act_precision) + 1, 3)))
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    # plt.show()

    plt.savefig(filename, dpi=300)
    plt.close(fig)