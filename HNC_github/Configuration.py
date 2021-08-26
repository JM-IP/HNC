#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-10-12 16:08
# @Author  : Jianming Ip
# @Site    : 
# @File    : Configuration.py
# @Company : VMC Lab in Peking University

import attr


@attr.s
class Configuration(object):
    experiment = attr.ib()
    optimizer = attr.ib(default=None)
    weightfile = attr.ib(default=None)

    w_quantize = attr.ib(default=None)
    a_quantize = attr.ib(default=None)

    # Uniform quantization (sign=True):
    #   xmax = stepsize * ( 2**(bitwidth-1) - 1 )
    #      -> xmax_min = stepsize_min * ( 2**(bitwidth_min-1) - 1)
    #      -> xmax_max = stepsize_max * ( 2**(bitwdith_max-1) - 1)
    # Pow2 quantization (sign=True, zero=True):
    #   xmax = xmin * 2**(2**(bitwidth-2) - 1)
    w_stepsize = attr.ib(converter=float, default=2**-3)
    w_stepsize_min = attr.ib(converter=float, default=2**-8)
    w_stepsize_max = attr.ib(converter=float, default=1**-1)
    w_xmin_min = attr.ib(converter=float, default=2**-16)
    w_xmin_max = attr.ib(converter=float, default=127/2.0)
    w_xmax_min = attr.ib(converter=float, default=2**-8)
    w_xmax_max = attr.ib(converter=float, default=127/2.0)
    w_bitwidth = attr.ib(converter=int, default=4)
    w_bitwidth_min = attr.ib(converter=int, default=2)  # one bit for sign
    w_bitwidth_max = attr.ib(converter=int, default=8)

    # Uniform quantization (sign=False):
    #   xmax = stepsize * ( 2**bitwidth - 1 )
    # Pow2 quantization (sign=False, zero=True)
    #   xmax = xmin * 2**(2**(bitwidth-1) - 1)
    a_stepsize = attr.ib(converter=float, default=2**-5)
    a_stepsize_min = attr.ib(converter=float, default=2**-8)
    a_stepsize_max = attr.ib(converter=float, default=1)
    a_xmin_min = attr.ib(converter=float, default=2**-14)
    a_xmin_max = attr.ib(converter=float, default=255/4.0)
    a_xmax_min = attr.ib(converter=float, default=2**-8)
    a_xmax_max = attr.ib(converter=float, default=255/4.0)
    a_bitwidth = attr.ib(converter=int, default=4)
    a_bitwidth_min = attr.ib(converter=int, default=1)
    a_bitwidth_max = attr.ib(converter=int, default=8)

    target_weight_kbytes = attr.ib(converter=float, default=-1)
    target_activation_kbytes = attr.ib(converter=float, default=-1)
    target_activation_type = attr.ib(default='max')

    initial_cost_lambda2 = attr.ib(converter=float, default=0.1)
    initial_cost_lambda3 = attr.ib(converter=float, default=0.1)
    no_bias = attr.ib(converter=bool, default=True)
    scale_layer = attr.ib(converter=bool, default=False)
    data_train = attr.ib(converter=str, default='CIFAR10')
    data_dir = attr.ib(converter=str, default='/home/yjm/D/yjm/projects/data/')
    train_batch_size = attr.ib(converter=int, default=256)
    eval_batch_size = attr.ib(converter=int, default=256)
    wd = attr.ib(converter=float, default=2e-4)
    kernel_size = attr.ib(converter=int, default=3)
    max_epochs = attr.ib(converter=int, default=300)
    n_colors = attr.ib(converter=int, default=3)
    downsample_type = attr.ib(converter=str, default='A')
    pretrain_dir = attr.ib(converter=str, default='')
    num_workers = attr.ib(converter=int, default=1)
    log_interval = attr.ib(converter=int, default=10)
    log_name = attr.ib(converter=str, default='')
