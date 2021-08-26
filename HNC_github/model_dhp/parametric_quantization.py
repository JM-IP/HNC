#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020-10-13 09:38
# @Author  : Jianming Ip
# @Site    : 
# @File    : parametric_quantization.py
# @Company : VMC Lab in Peking University

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def RoundNoGradientfunc():
    class RoundNoGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input.round()
        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return RoundNoGradient().apply


def CeilNoGradientfunc():
    class CeilNoGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input.ceil()
        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            return grad_input
    return CeilNoGradient().apply


def ClampNoGradientfunc(x_max, x_min):
    class ClampNoGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.clamp(max=x_max, min=x_min)
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input > x_max] = 0
            grad_input[input < x_min] = 0
            return grad_input
    return ClampNoGradient().apply


def SignNoGradientfunc():
    class SignNoGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (input.sign()+0.0001).sign()
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            return grad_input
    return SignNoGradient().apply
# def uniform_quantize(input, x_max, x_min, d_q):
#     try:
#         x_max = x_max.item()
#         x_min = x_min.item()
#     except:
#         pass
#     myclamp = ClampNoGradientfunc(x_max, x_min)
#     myround = RoundNoGradientfunc()
#     out = myround(myclamp(input) / d_q) * d_q
#     return out


def clip_scalar(v, min_value, max_value):
    myclamp = ClampNoGradientfunc(max_value, min_value)
    return myclamp(v)


def quantize_pow2(v):
    myround = RoundNoGradientfunc()
    return 2 ** (myround(torch.log(v) / 0.6931471805599453))


class fixed_point_quantize(nn.Module):
    def __init__(self, sign=True, k=8, delta=2**-4, fix_parameters=False):
        super(fixed_point_quantize, self).__init__()
        self.sign = sign
        self.k = k
        self.delta = delta
        if self.sign:
            xmax = float(2 ** (self.k - 1) - 1) * self.delta
            xmin = -xmax
        else:
            n = float(2 ** self.k - 1)
            xmax = n * self.delta
            xmin = 0
        self.xmin = xmin
        self.xmax = xmax
        self.uniform_q = uniform_quantize(self.xmax, self.xmin, self.delta)
    def forward(self, x):
        return self.uniform_q(x)


class parametric_fixed_point_quantize(nn.Module):
    def __init__(self, sign=True, n_init=8,
                 n_min=2, n_max=16, m_init=1,
                 m_min=-8, m_max=8, fix_parameters=False):
        super(parametric_fixed_point_quantize, self).__init__()
        self.sign = sign
        self.n_min = n_min
        self.n_max = n_max
        self.m_min = m_min
        self.m_max = m_max
        self.n = torch.nn.Parameter(Variable(torch.Tensor([n_init])), requires_grad=True)
        self.m = torch.nn.Parameter(Variable(torch.Tensor([m_init])), requires_grad=True)

    def forward(self, x):
        n_q = torch.round(clip_scalar(self.n, self.n_min, self.n_max))
        if self.sign:
            n_q = n_q - 1

        # ensure that dynamic range is in specified range
        m_q = clip_scalar(self.m, self.m_min, self.m_max)

        # compute step size from dynamic range and make sure that it is a pow2
        d_q = quantize_pow2((2 ** m_q) / (2 ** n_q - 1))

        # compute min/max value that we can represent
        x_max = d_q * (2 ** n_q - 1)
        if self.sign:
            x_min = -x_max
        else:
            x_min = 0
            # apply fixed-point quantization
        uniform_q = uniform_quantize(x_max, x_min, d_q)
        return uniform_q(x)


# CASE A: PARAMETRIZATION BY B AND XMAX
class parametric_fixed_point_quantize_b_xmax(nn.Module):
    def __init__(self, sign=True, n_init=8, n_min=2,
                 n_max=16, xmax_init=1, xmax_min=0.001,
                 xmax_max=10, fix_parameters=False):
        super(parametric_fixed_point_quantize_b_xmax, self).__init__()
        self.sign = sign
        self.n_min = n_min
        self.n_max = n_max
        self.xmax_min = xmax_min
        self.xmax_max = xmax_max
        self.n = torch.nn.Parameter(Variable(torch.Tensor([n_init])),
                                    requires_grad=True)
        self.xmax = torch.nn.Parameter(Variable(torch.Tensor([xmax_init])),
                                       requires_grad=True)
        self.xmin = Variable(torch.Tensor([0]), requires_grad=False).cuda()

    def forward(self, x):
        # ensure that bitwidth is in specified range and an integer
        myround = RoundNoGradientfunc()
        n = myround(clip_scalar(self.n, self.n_min, self.n_max))
        if self.sign:
            n = n - 1
        # ensure that dynamic range is in specified range
        xmax = clip_scalar(self.xmax, self.xmax_min, self.xmax_max)

        # compute step size from dynamic range and make sure that it is a pow2
        d = quantize_pow2(xmax / (2 ** n - 1))

        # compute min/max value that we can represent
        if self.sign:
            self.xmin = -xmax
        xmin = self.xmin
        xmax=xmax.repeat(x.shape)
        xmin=xmin.repeat(x.shape)
        x = torch.min(xmax, x)
        x = torch.max(xmin, x)

        myround = RoundNoGradientfunc()
        out = d * myround(x/ d)
        return out #uniform_q(x)

# CASE B: PARAMETRIZATION BY D AND XMAX
class parametric_fixed_point_quantize_d_xmax(nn.Module):
    def __init__(self, sign=True, d_init=2**-4,
                 d_min=2**-8, d_max=2**8,
                 xmax_init=1, xmax_min=0.001,
                 xmax_max=10, fix_parameters=False):
        super(parametric_fixed_point_quantize_d_xmax, self).__init__()
        self.sign = sign
        self.d_min = d_min
        self.d_max = d_max
        self.xmax_min = xmax_min
        self.xmax_max = xmax_max
        self.d = torch.nn.Parameter(Variable(torch.Tensor([d_init])),
                                    requires_grad=not fix_parameters)
        self.xmax = torch.nn.Parameter(Variable(torch.Tensor([xmax_init])),
                                       requires_grad=not fix_parameters)
        if self.sign:
            self.xmin = - self.xmax
        else:
            self.xmin = torch.nn.Parameter(Variable(torch.Tensor([0])),
                                       requires_grad=False)

    def forward(self, x):
        # ensure that stepsize is in specified range and a power of two
        d = quantize_pow2(clip_scalar(self.d, self.d_min, self.d_max))

        # ensure that dynamic range is in specified range
        xmax = clip_scalar(self.xmax, self.xmax_min, self.xmax_max)

        # compute min/max value that we can represent

        # apply fixed-point quantization
        if self.sign:
            self.xmin = - self.xmax
        xmin = self.xmin
        xmax = xmax.repeat(x.shape)
        xmin = xmin.repeat(x.shape)
        x = torch.min(xmax, x)
        x = torch.max(xmin, x)

        myround = RoundNoGradientfunc()
        out = d * myround(x/ d)
        return out #uniform_q(x)
        # uniform_q = uniform_quantize(x, xmax, xmin, d)
        # return uniform_q


# CASE C: PARAMETRIZATION BY B AND D
class parametric_fixed_point_quantize_d_b(nn.Module):
    def __init__(self, sign=True, n_init=8, n_min=2,
                 n_max=16, d_init=2**-4, d_min=2**-8,
                 d_max=2**8, fix_parameters=False):
        super(parametric_fixed_point_quantize_d_b, self).__init__()
        self.sign = sign
        self.n_min = n_min
        self.n_max = n_max
        self.d_min = d_min
        self.d_max = d_max
        self.n = torch.nn.Parameter(Variable(torch.Tensor([n_init])), requires_grad=True)
        self.d = torch.nn.Parameter(Variable(torch.Tensor([d_init])), requires_grad=True)
        self.xmin = Variable(torch.Tensor([0]), requires_grad=False).cuda()

    def forward(self, x):
        # ensure that bitwidth is in specified range and an integer
        n = torch.round(clip_scalar(self.n, self.n_min, self.n_max))
        if self.sign:
            n = n - 1

        # ensure that stepsize is in specified range and a power of two
        d = quantize_pow2(clip_scalar(self.d, self.d_min, self.d_max))

        # ensure that dynamic range is in specified range
        xmax = d * (2 ** n - 1)

        # compute min/max value that we can represent
        if self.sign:
            self.xmin = -xmax

        xmin = self.xmin

        # apply fixed-point quantization
        xmax = xmax.repeat(x.shape)
        xmin = xmin.repeat(x.shape)
        x = torch.min(xmax, x)
        x = torch.max(xmin, x)

        myround = RoundNoGradientfunc()
        out = d * myround(x/ d)
        return out #uniform_q(x)
        # uniform_q = uniform_quantize(x, xmax, xmin, d)
        # return uniform_q


def linear_Q_fn(cfg=None):
    class Linear_Q(nn.Linear):
        def __init__(self, in_features, out_features, bias=False):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.cfg = cfg
            self.bias_on = bias
            stdv = 1. / np.sqrt(np.prod(self.weight.shape))
            torch.nn.init.uniform_(self.weight, a=-stdv, b=stdv)
            if bias:
                self.quantize_w, self.quantize_b = get_quantizers(cfg=self.cfg, w=self.weight, delta_init=cfg.w_stepsize)
            else:
                self.quantize_w, _ = get_quantizers(cfg=self.cfg, w=self.weight, delta_init=cfg.w_stepsize)



        def forward(self, input):
            # self.quantize_w, self.quantize_b = get_quantizers(self.weight, cfg=self.cfg)

            if self.quantize_w is not None:
                weight_q = self.quantize_w(self.weight)
            else:
                weight_q = self.weight
            if self.bias is not None:
                if self.quantize_b is not None:
                    self.bias = self.quantize_b(self.bias)
            return F.linear(input, weight_q, self.bias)

    return Linear_Q


def conv2d_Q_fn(cfg=None):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
            self.cfg = cfg
            torch.nn.init.xavier_uniform(self.weight)
            if bias:
                self.quantize_w, self.quantize_b = get_quantizers(w=self.weight, cfg=self.cfg, delta_init=cfg.w_stepsize)
            else:
                self.quantize_w, _ = get_quantizers(w=self.weight, cfg=self.cfg, delta_init=cfg.w_stepsize)


        def forward(self, input):
            # self.quantize_w, self.quantize_b = get_quantizers(self.weight, cfg=self.cfg)

            if self.quantize_w is not None:
                weight_q = self.quantize_w(self.weight)
            else:
                weight_q = self.weight
            if self.bias is not None:
                if self.quantize_b is not None:
                    bias = self.quantize_b(self.bias)
            else:
                bias = self.bias

            return F.conv2d(input, weight_q, bias, self.stride,
                            self.padding, self.dilation, self.groups)

    return Conv2d_Q


def activation_Q_fn(cfg):
    class activation_quantize_fn(nn.Module):
        def __init__(self, inplace=False, fix_parameters=False):
            super(activation_quantize_fn, self).__init__()
            self.a_stepsize = cfg.a_stepsize
            self.a_stepsize_min = cfg.a_stepsize_min
            self.a_stepsize_max = cfg.a_stepsize_max
            self.a_bitwidth = cfg.a_bitwidth
            self.a_bitwidth_max = cfg.a_bitwidth_max
            self.a_bitwidth_min = cfg.a_bitwidth_min
            self.a_xmin_min = cfg.a_xmin_min
            self.a_xmin_max = cfg.a_xmin_max
            self.a_xmax_min = cfg.a_xmax_min
            self.a_xmax_max = cfg.a_xmax_max
            self.a_quantize = cfg.a_quantize
            self.inplace = inplace
            self.delta = self.a_stepsize
            self.a_size = torch.nn.Parameter(Variable(torch.Tensor([0])), requires_grad=False)
            self.fix_parameters = fix_parameters
            xmax = self.delta * (2. ** self.a_bitwidth - 1)

            if self.a_quantize is not None and 'pow2' in self.a_quantize:
                xmax = 2. ** np.round(np.log2(xmax))
                xmin = xmax / 2. ** (2. ** (self.a_bitwidth - 1) - 1)

                xmin = np.clip(xmin, self.a_xmin_min + 1e-5, self.a_xmin_max - 1e-5)
                xmax = np.clip(xmax, self.a_xmax_min + 1e-5, self.a_xmax_max - 1e-5)

                self.xmin = xmin

            self.xmax = xmax
            if self.a_quantize is not None:
                print(f'We use default delta ({self.delta, xmax}) for quantized nonlinearity.')
            self.a_quant = self.choose_quan_func()

        def choose_quan_func(self):
            if self.a_quantize == 'fp_relu':
                return fixed_point_quantize(sign=False, k=self.a_bitwidth, delta=self.a_stepsize)
            elif self.a_quantize == 'parametric_fp_b_xmax_relu':
                return parametric_fixed_point_quantize_b_xmax(sign=False, n_init=self.a_bitwidth,
                                                              n_min=self.a_bitwidth_min, n_max=self.a_bitwidth_max,
                                                              xmax_init=self.xmax, xmax_min=self.a_xmax_min,
                                                              xmax_max=self.a_xmax_max)
            elif self.a_quantize == 'parametric_fp_d_xmax_relu':
                return parametric_fixed_point_quantize_d_xmax(sign=False,
                                                              d_init=self.delta,
                                                              d_min=self.a_stepsize_min,
                                                              d_max=self.a_stepsize_max,
                                                              xmax_init=self.xmax,
                                                              xmax_min=self.a_xmax_min,
                                                              xmax_max=self.a_xmax_max,
                                                              fix_parameters=self.fix_parameters)
            elif self.a_quantize == 'parametric_fp_d_b_relu':
                return parametric_fixed_point_quantize_d_b(sign=False,
                                                           n_init=self.a_bitwidth,
                                                           n_min=self.a_bitwidth_min,
                                                           n_max=self.a_bitwidth_max,
                                                           d_init=self.delta,
                                                           d_min=self.a_stepsize_min,
                                                           d_max=self.a_stepsize_max)
            else:
                return nn.ReLU(inplace=False)
            
        def forward(self, x):
            if self.a_size.item()==0:
                self.a_size.data = Variable(torch.Tensor([np.prod(x.shape[1::])]), requires_grad=False).cuda()
            return self.a_quant(x)

    return activation_quantize_fn


def find_delta(w, bw):
    """ Finds optimal quantization step size for FP quantization """
    maxabs_w = np.max(np.abs(w.cpu().detach().cpu().numpy())) + np.finfo(np.float32).eps

    if bw > 4:
        return 2**(np.ceil(np.log2(maxabs_w/(2**(bw-1)-1))))
    else:
        return 2**(np.floor(np.log2(maxabs_w/(2**(bw-1)-1))))


def get_quantizers(cfg, w=None, with_bias=True, delta_init=None, fix_parameters=False):
    if cfg.w_quantize is None:
        return None, None
    if delta_init is not None or w is None:
        delta = delta_init
    else:
        print("yjm is here: looking for suitable delta")
        delta = find_delta(w, cfg.w_bitwidth)
    xmax = delta * (2 ** (cfg.w_bitwidth - 1) - 1)
    if 'pow2' in cfg.w_quantize:
        xmax = 2. ** np.round(np.log2(xmax))
        xmin = xmax / 2. ** (2. ** (cfg.w_bitwidth - 1) - 1)

        xmin = np.clip(xmin, cfg.w_xmin_min + 1e-5, cfg.w_xmin_max - 1e-5)
        xmax = np.clip(xmax, cfg.w_xmax_min + 1e-5, cfg.w_xmax_max - 1e-5)

    print(f'Quantized affine/conv initialized to delta={delta}, xmax={xmax}', end='')
    if w is not None:
        print(f', weight shape is {w.shape}')
    else:
        print(f'')
    quantization_b = None
    if cfg.w_quantize == 'fp':
        quantization_w = fixed_point_quantize(sign=True, k=cfg.w_bitwidth, delta=delta)
        if with_bias:
            quantization_b = fixed_point_quantize(sign=True, k=cfg.w_bitwidth, delta=delta)
    elif cfg.w_quantize == 'parametric_fp_b_xmax':
        quantization_w = \
            parametric_fixed_point_quantize_b_xmax(sign=True, n_init=cfg.w_bitwidth,
                                                   n_min=cfg.w_bitwidth_min,
                                                   n_max=cfg.w_bitwidth_max,
                                                   xmax_init=xmax,
                                                   xmax_min=cfg.w_xmax_min,
                                                   xmax_max=cfg.w_xmax_max)
        if with_bias:
            quantization_b = parametric_fixed_point_quantize_b_xmax(n_init=cfg.w_bitwidth,
                                     n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                     xmax_init=xmax,
                                     xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max)
    elif cfg.w_quantize == 'parametric_fp_d_xmax':
        quantization_w = parametric_fixed_point_quantize_d_xmax(sign=True,
                                 d_init=delta,
                                 d_min=cfg.w_stepsize_min, d_max=cfg.w_stepsize_max,
                                 xmax_init=xmax,
                                 xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                 fix_parameters=fix_parameters)
        if with_bias:
            quantization_b = parametric_fixed_point_quantize_d_xmax(sign=True,
                                 d_init=delta,
                                 d_min=cfg.w_stepsize_min, d_max=cfg.w_stepsize_max,
                                 xmax_init=xmax,
                                 xmax_min=cfg.w_xmax_min, xmax_max=cfg.w_xmax_max,
                                 fix_parameters=fix_parameters)
    elif cfg.w_quantize == 'parametric_fp_d_b':
        quantization_w = parametric_fixed_point_quantize_d_b(sign=True,
                                 n_init=cfg.w_bitwidth,
                                 n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                 d_init=delta,
                                 d_min=cfg.w_stepsize_min, d_max=cfg.w_stepsize_max)
        if with_bias:
            quantization_b = parametric_fixed_point_quantize_d_b(sign=True,
                                 n_init=cfg.w_bitwidth,
                                 n_min=cfg.w_bitwidth_min, n_max=cfg.w_bitwidth_max,
                                 d_init=delta,
                                 d_min=cfg.w_stepsize_min, d_max=cfg.w_stepsize_max)
    else:
        quantization_w = None
        quantization_b = None

    return quantization_w, quantization_b


def network_size_weights(model, cfg):
    """
    Return total number of weights and network size (for weights) in KBytes
    """
    num_params=0
    kbytes=0
    mydict = dict(model.named_parameters())
    n_dict = dict()
    for name, v in model.named_modules():
        if isinstance(v, torch.nn.Conv2d):
            value = mydict[name+'.weight']
            _num_params = np.prod(value.shape)
            if cfg.w_quantize is not None:
                if cfg.w_quantize in ['parametric_fp_b_xmax',
                                      'parametric_fp_d_b',
                                      'parametric_pow2_b_xmax',
                                      'parametric_pow2_b_xmin']:
                    n = RoundNoGradientfunc()(clip_scalar(mydict[name+'.quantize_w.n'], cfg.w_bitwidth_min, cfg.w_bitwidth_max))
                elif cfg.w_quantize == 'parametric_fp_d_xmax':
                    d = mydict[name+'.quantize_w.d']
                    xmax = mydict[name+'.quantize_w.xmax']

                    # ensure that stepsize is in specified range and a power of two
                    d_q = quantize_pow2(clip_scalar(d, cfg.w_stepsize_min, cfg.w_stepsize_max))

                    # ensure that dynamic range is in specified range
                    xmax = clip_scalar(xmax, cfg.w_xmax_min, cfg.w_xmax_max)

                    # compute real `xmax`
                    xmax = RoundNoGradientfunc()(xmax / d_q) * d_q

                    # we do not clip to `cfg.w_bitwidth_max` as xmax/d_q could correspond to more than 8 bit

                    n = torch.max(CeilNoGradientfunc()(torch.log(xmax/d_q + 1.0)/0.6931471805599453 + 1.0),
                                  Variable(torch.Tensor([cfg.w_bitwidth_min]).cuda(), requires_grad=False))
                elif cfg.w_quantize == 'parametric_pow2_xmin_xmax':
                    xmin = mydict[name+'.quantize_w.xmin']
                    xmax = mydict[name+'.quantize_w.xmax']
                    # ensure that minimum dynamic range is in specified range and a power-of-two
                    xmin = quantize_pow2(clip_scalar(xmin, cfg.w_xmin_min, cfg.w_xmin_max))
                    # ensure that maximum dynamic range is in specified range and a power-of-two
                    xmax = quantize_pow2(clip_scalar(xmax, cfg.w_xmax_min, cfg.w_xmax_max))
                    n = torch.max(CeilNoGradientfunc()(torch.log(torch.log(xmax/xmin)/0.6931471805599453 + 1.0)/0.6931471805599453 + 1.), cfg.w_bitwidth_min)

                elif cfg.w_quantize == 'fp' or cfg.w_quantize == 'pow2':
                    n = cfg.w_bitwidth
                else:
                    raise ValueError(f'Unknown quantization method {cfg.w_quantize}')
            else:
                # float precision
                n = 32
            n_dict[name] = n
            # print('layer '+name+' is %d bits'%n)

            kbytes += n * _num_params / 8. / 1024.
            num_params += _num_params
    return kbytes, num_params, n_dict


def network_size_activations(model, cfg):
    kbytes = []
    num_activations = 0
    mydict = dict(model.named_parameters())
    n_dict = dict()
    for name, value in model.named_parameters():
        #     print(name)
        if '.a_size' in name:
            num_activations += value.item()
            if cfg.a_quantize is not None:
                if cfg.a_quantize in ['fp_relu', 'pow2_relu']:
                    n = cfg.a_bitwidth
                elif cfg.a_quantize in ['parametric_fp_relu',
                                        'parametric_fp_b_xmax_relu',
                                        'parametric_fp_d_b_relu',
                                        'parametric_pow2_b_xmax_relu',
                                        'parametric_pow2_b_xmin_relu']:
                    n = RoundNoGradientfunc()(clip_scalar(mydict[name.replace('.a_size', '.a_quant.n')],
                                                          cfg.a_bitwidth_min, cfg.a_bitwidth_max))
                elif cfg.a_quantize in ['parametric_fp_d_xmax_relu']:
                    d = mydict[name.replace('.a_size', '.a_quant.d')]
                    xmax = mydict[name.replace('.a_size', '.a_quant.xmax')]

                    # ensure that stepsize is in specified range and a power of two
                    d_q = quantize_pow2(clip_scalar(d, cfg.a_stepsize_min, cfg.a_stepsize_max))

                    # ensure that dynamic range is in specified range
                    xmax = clip_scalar(xmax, cfg.a_xmax_min, cfg.a_xmax_max)

                    # compute real `xmax`
                    xmax = RoundNoGradientfunc()(xmax / d_q) * d_q
                    n = torch.max(CeilNoGradientfunc()(torch.log(xmax/d_q + 1.0)/0.6931471805599453),
                                  Variable(torch.Tensor([cfg.a_bitwidth_min]).cuda(), requires_grad=False))
                elif cfg.a_quantize in ['parametric_pow2_xmin_xmax_relu']:
                    xmin = mydict[name.replace('.a_size', '.a_quant.xmin')]
                    xmax = mydict[name.replace('.a_size', '.a_quant.xmax')]
                    xmin = quantize_pow2(clip_scalar(xmin, cfg.a_xmin_min, cfg.a_xmin_max))
                    xmax = quantize_pow2(clip_scalar(xmax, cfg.a_xmax_min, cfg.a_xmax_max))
                    n = torch.max(CeilNoGradientfunc()(torch.log(torch.log(xmax/xmin)/0.6931471805599453 + 1.)/0.6931471805599453 + 1.), Variable(torch.Tensor([cfg.w_bitwidth_min]).cuda(), requires_grad=False))
                else:
                    raise ValueError("Unknown quantization method {}".format(cfg.a_quantize))
            else:
                # float precision
                n = Variable(torch.Tensor([32]).cuda(), requires_grad=False)
            n_dict[name] = n
            kbytes.append(n * num_activations / 8. / 1024.)

    if cfg.target_activation_type == 'max':
        _kbytes = torch.max(torch.cat(kbytes, dim=0), dim=0)[0]
    elif cfg.target_activation_type == 'sum':
        _kbytes = torch.sum(torch.cat(kbytes, dim=0), dim=0)
    return _kbytes, num_activations, n_dict


def clip_quant_grads(model, cfg):
    mydict = dict(model.named_parameters())
    for name, value in model.named_parameters():
        if 'quantize_w.xmax' in name:
            if cfg.w_quantize == 'parametric_fp_d_xmax':
                xmax = value
                d = mydict[name.replace('quantize_w.xmax', 'quantize_w.d')]
                d.grad = d.grad.clamp(-d.item(), d.item())
                xmax.grad = xmax.grad.clamp(-d.item(), d.item())
            elif cfg.w_quantize == 'parametric_pow2_xmin_xmax':
                xmax = value
                xmin = mydict[name.replace('quantize_w.xmax', 'quantize_w.min')]
                xmin.grad = xmin.grad.clamp(-xmin.item(), xmin.item())
                xmax.grad = xmax.grad.clamp(-xmin.item(), xmin.item())
        if 'a_quant.xmax' in name:
            if cfg.a_quantize == 'parametric_fp_d_xmax_relu':
                xmax = value
                d = mydict[name.replace('a_quant.xmax', 'a_quant.d')]

                d.grad = d.grad.clamp(-d.item(), d.item())
                xmax.grad = xmax.grad.clamp(-d.item(), d.item())

            elif cfg.a_quantize == 'parametric_pow2_xmin_xmax_relu':
                xmax = value
                xmin = mydict[name.replace('a_quant.xmax', 'a_quant.xmin')]

                xmin.grad = xmin.grad.clamp(-xmin.item(), xmin.item())
                xmax.grad = xmax.grad.clamp(-xmin.item(), xmin.item())


def clip_quant_vals(model, cfg):
    mydict = dict(model.named_parameters())
    for name, value in model.named_parameters():
        if cfg.w_quantize in ['parametric_fp_b_xmax',
                              'parametric_fp_d_xmax',
                              'parametric_fp_d_b',
                              'parametric_pow2_b_xmax',
                              'parametric_pow2_b_xmin',
                              'parametric_pow2_xmin_xmax']:
            if 'quantize_w.m' in name:
                value.data = value.data.clamp(cfg.w_dynrange_min + 1e-5, cfg.w_dynrange_max - 1e-5)
            elif 'quantize_w.n' in name:
                value.data = value.data.clamp(cfg.w_bitwidth_min + 1e-5, cfg.w_bitwidth_max - 1e-5)
            elif 'quantize_w.d' in name:
                if cfg.w_quantize == 'parametric_fp_d_xmax':
                    xmax = mydict[name.replace('quantize_w.d', 'quantize_w.xmax')]
                    min_value = min(value.data, xmax.data - 1e-5)
                    max_value = max(value.data + 1e-5, xmax.data)
                    value.data = min_value
                    xmax.data = max_value
                value.data = value.data.clamp(cfg.w_stepsize_min + 1e-5, cfg.w_stepsize_max - 1e-5)
            elif 'quantize_w.xmax' in name:
                value.data = value.data.clamp(cfg.w_xmax_min + 1e-5, cfg.w_xmax_max - 1e-5)
            elif 'quantize_w.xmin' in name:
                if cfg.w_quantize == 'parametric_pow2_xmin_xmax':
                    xmax = mydict[name.replace('quantize_w.xmin', 'quantize_w.xmax')]
                    min_value = min(value.data, xmax.data + 1e-5)
                    max_value = max(value.data + 1e-5, xmax.data)
                    value.data = min_value
                    xmax.data = max_value
                value.data = value.data.clamp(cfg.w_xmin_min + 1e-5, cfg.w_xmin_max - 1e-5)

        if cfg.a_quantize in ['parametric_fp_b_xmax_relu',
                          'parametric_fp_d_xmax_relu',
                          'parametric_fp_d_b_relu',
                          'parametric_pow2_b_xmax_relu',
                          'parametric_pow2_b_xmin_relu',
                          'parametric_pow2_xmin_xmax_relu']:
            if 'a_quant.m' in name:
                value.data = value.data.clamp(cfg.a_dynrange_min + 1e-5, cfg.a_dynrange_max - 1e-5)
            if 'a_quant.n' in name:
                value.data = value.data.clamp(cfg.a_bitwidth_min + 1e-5, cfg.a_bitwidth_max - 1e-5)
            if 'a_quant.d' in name:
                if cfg.a_quantize == 'parametric_fp_d_xmax_relu':
                    xmax = mydict[name.replace('a_quant.d', 'a_quant.xmax')]
                    min_value = min(value.data, xmax.data - 1e-5)
                    max_value = max(value.data + 1e-5, xmax.data)
                    value.data = min_value
                    xmax.data = max_value
                value.data = value.data.clamp(cfg.a_stepsize_min + 1e-5, cfg.a_stepsize_max - 1e-5)
            if 'a_quant.xmax' in name:
                value.data = value.data.clamp(cfg.a_xmax_min + 1e-5, cfg.a_xmax_max - 1e-5)
            if 'a_quant.xmin' in name:
                if cfg.a_quantize == 'parametric_pow2_xmin_xmax_relu':
                    xmax = mydict[name.replace('a_quant.xmin', 'a_quant.xmax')]
                    min_value = min(value.data, xmax.data + 1e-5)
                    max_value = max(value.data + 1e-5, xmax.data)
                    value.data = min_value
                    xmax.data = max_value
                value.data = value.data.clamp(cfg.a_xmin_min + 1e-5, cfg.a_xmin_max - 1e-5)

def get_percision(quantize_w, cfg=None):
    if cfg.w_quantize is not None:
        if cfg.w_quantize in ['parametric_fp_b_xmax',
                              'parametric_fp_d_b',
                              'parametric_pow2_b_xmax',
                              'parametric_pow2_b_xmin']:
            n = RoundNoGradientfunc()(clip_scalar(quantize_w.n, cfg.w_bitwidth_min, cfg.w_bitwidth_max))
        elif cfg.w_quantize == 'parametric_fp_d_xmax':
            d = quantize_w.d
            xmax = quantize_w.xmax

            # ensure that stepsize is in specified range and a power of two
            d_q = quantize_pow2(clip_scalar(d, cfg.w_stepsize_min, cfg.w_stepsize_max))

            # ensure that dynamic range is in specified range
            xmax = clip_scalar(xmax, cfg.w_xmax_min, cfg.w_xmax_max)

            # compute real `xmax`
            xmax = RoundNoGradientfunc()(xmax / d_q) * d_q

            # we do not clip to `cfg.w_bitwidth_max` as xmax/d_q could correspond to more than 8 bit

            n = torch.max(CeilNoGradientfunc()(torch.log(xmax/d_q + 1.0)/0.6931471805599453 + 1.0),
                          Variable(torch.Tensor([cfg.w_bitwidth_min]).cuda(), requires_grad=False))
        elif cfg.w_quantize == 'parametric_pow2_xmin_xmax':
            xmin = quantize_w.xmin
            xmax = quantize_w.xmax
            # ensure that minimum dynamic range is in specified range and a power-of-two
            xmin = quantize_pow2(clip_scalar(xmin, cfg.w_xmin_min, cfg.w_xmin_max))
            # ensure that maximum dynamic range is in specified range and a power-of-two
            xmax = quantize_pow2(clip_scalar(xmax, cfg.w_xmax_min, cfg.w_xmax_max))
            n = torch.max(CeilNoGradientfunc()(torch.log(torch.log(xmax/xmin)/0.6931471805599453 + 1.0)/0.6931471805599453 + 1.), cfg.w_bitwidth_min)

        elif cfg.w_quantize == 'fp' or cfg.w_quantize == 'pow2':
            n = cfg.w_bitwidth
        else:
            raise ValueError(f'Unknown quantization method {cfg.w_quantize}')
    else:
        # float precision
        n = Variable(torch.Tensor([32]).cuda(), requires_grad=False)
    return n

def get_percision_a(quantize_a, cfg=None):
    # TODO:: fix this
    if cfg.a_quantize is not None:
        if cfg.a_quantize in ['parametric_fp_relu',
                              'parametric_fp_b_xmax_relu',
                              'parametric_fp_d_b_relu',
                              'parametric_pow2_b_xmax_relu',
                              'parametric_pow2_b_xmin_relu']:
            n = RoundNoGradientfunc()(clip_scalar(quantize_a.n, cfg.a_bitwidth_min, cfg.a_bitwidth_max))
        elif cfg.a_quantize == 'parametric_fp_d_xmax_relu':
            d = quantize_a.d
            xmax = quantize_a.xmax

            # ensure that stepsize is in specified range and a power of two
            d_q = quantize_pow2(clip_scalar(d, cfg.a_stepsize_min, cfg.a_stepsize_max))

            # ensure that dynamic range is in specified range
            xmax = clip_scalar(xmax, cfg.a_xmax_min, cfg.a_xmax_max)

            # compute real `xmax`
            xmax = RoundNoGradientfunc()(xmax / d_q) * d_q

            # we do not clip to `cfg.w_bitwidth_max` as xmax/d_q could correspond to more than 8 bit

            n = torch.max(CeilNoGradientfunc()(torch.log(xmax/d_q + 1.0)/0.6931471805599453),
                          Variable(torch.Tensor([cfg.a_bitwidth_min]).cuda(), requires_grad=False))
        elif cfg.a_quantize == 'parametric_pow2_xmin_xmax_relu':
            # xmin = quantize_w.xmin
            # xmax = quantize_w.xmax
            # # ensure that minimum dynamic range is in specified range and a power-of-two
            # xmin = quantize_pow2(clip_scalar(xmin, cfg.w_xmin_min, cfg.w_xmin_max))
            # # ensure that maximum dynamic range is in specified range and a power-of-two
            # xmax = quantize_pow2(clip_scalar(xmax, cfg.w_xmax_min, cfg.w_xmax_max))
            # n = torch.max(CeilNoGradientfunc()(torch.log(torch.log(xmax/xmin)/0.6931471805599453 + 1.0)/0.6931471805599453 + 1.), cfg.w_bitwidth_min)
            raise ValueError(f'Unknown quantization method {cfg.a_quantize}')
        elif cfg.a_quantize == 'fp' or cfg.a_quantize == 'pow2':
            n = cfg.a_bitwidth
        else:
            raise ValueError(f'Unknown quantization method {cfg.w_quantize}')
    else:
        # float precision
        n = Variable(torch.Tensor([32]).cuda(), requires_grad=False)
    return n
    pass