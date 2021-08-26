import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import model_dhp.parametric_quantization as PQ

def make_model(args, cfg):
    return ResNet(args,cfg)

class PQConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 bias=False, groups=1, cfg=None, args=None, quant=True):
        super(PQConv, self).__init__(in_channels, out_channels, kernel_size, bias=bias)
        self.weight = nn.Parameter(torch.rand((out_channels, in_channels, kernel_size, kernel_size)) * 0.001, requires_grad=True)
        self.stride = [stride, stride]
        self.groups = groups
        self.kernel_size = [kernel_size, kernel_size]
        self.padding = [self.kernel_size[0]//2, self.kernel_size[1]//2]
        self.cfg = cfg
        self.name='conv'
        if quant:
            # if bias:
            #     self.quantize_w, self.quantize_b = PQ.get_quantizers(w=None, cfg=self.cfg, delta_init=cfg.w_stepsize)
            # else:
            self.quantize_w, _ = PQ.get_quantizers(w=None, cfg=self.cfg, delta_init=cfg.w_stepsize)
        else:
            self.quantize_w = None
            # self.quantize_b = None
        # if self.act:
        #     self.a_quant = PQ.activation_Q_fn(cfg)()
        # if self.args.cal_channel=='sign':
        #     self.calnumlayer=PQ.SignNoGradientfunc()
        # elif self.args.cal_channel=='sigmoid':
        #     self.calnumlayer=torch.nn.Sigmoid()
        # else:
        #     raise()
    def forward(self, x, getq=False):
        weight = self.weight
        if self.quantize_w is not None:
            weight = self.quantize_w(self.weight)
        # bias = self.bias
        # if self.bias is not None and self.quantize_b is not None:
        #     bias = self.quantize_b(self.bias)
        out = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv=PQConv, downsample=None, args=None, cfg=None, qact=True):
        super(ResBlock, self).__init__()

        self.stride = stride
        self.cfg = cfg
        self.a_quant = PQ.activation_Q_fn(cfg)()
        m = [conv(in_channels, planes, kernel_size, cfg=self.cfg, stride=stride, bias=False, args=args),
             nn.BatchNorm2d(planes),
             PQ.activation_Q_fn(self.cfg)(),
             # nn.ReLU(inplace=True),
             conv(planes, planes, kernel_size, cfg=self.cfg, bias=False, args=args),
             nn.BatchNorm2d(planes)]

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        # self.act_out = nn.ReLU(inplace=True)
        if qact:
            self.act_out = PQ.activation_Q_fn(self.cfg)()
        else:
            self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out

class ResNet(nn.Module):
    def __init__(self, args, cfg=None):
        super(ResNet, self).__init__()

        self.width_mult = args.width_mult
        self.cfg = cfg
        self.args = args
        if args.depth == 18:
            self.expansion = 1
            self.block = ResBlock
            self.n_blocks = [2, 2, 2, 2]
        else:
            raise()
        self.in_channels = int(64 * self.width_mult)
        # bias = not args.no_bias

        m = [PQConv(args.n_colors, int(64 * self.width_mult), 7, stride=2, cfg=self.cfg, bias=False, args=args),
             nn.BatchNorm2d(int(64 * self.width_mult)),
             PQ.activation_Q_fn(self.cfg)(),
             # nn.ReLU(inplace=True),
             nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
             self.make_layer(self.n_blocks[0], int(64 * self.width_mult), 3),
             self.make_layer(self.n_blocks[1], int(128 * self.width_mult), 3, stride=2),
             self.make_layer(self.n_blocks[2], int(256 * self.width_mult), 3, stride=2),
             self.make_layer(self.n_blocks[3], int(512 * self.width_mult), 3, stride=2, qact=False),
             nn.AdaptiveAvgPool2d((1, 1))]
        fc = nn.Linear(int(512 * self.width_mult) * self.expansion, 1000)

        self.features = nn.Sequential(*m)
        self.classifier = fc

    def make_layer(self, blocks, planes, kernel_size, stride=1, conv=PQConv, qact=True):
        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(*[conv(self.in_channels, out_channels, 1, stride=stride, cfg=self.cfg, args=self.args), nn.BatchNorm2d(out_channels)])
        else:
            downsample = None

        m = [self.block(self.in_channels, planes, kernel_size, stride=stride, cfg=self.cfg, downsample=downsample, qact=True)]
        self.in_channels = out_channels
        for i in range(blocks - 1):
            if i ==blocks - 2:
                m.append(self.block(self.in_channels, planes, kernel_size, cfg=self.cfg, qact=qact))
            else:
                m.append(self.block(self.in_channels, planes, kernel_size, cfg=self.cfg, qact=True))

        return nn.Sequential(*m)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.squeeze())
        return x

    def my_load(self, model):
        bn_layer_state_dict=[]
        conv_layer_state_dict=[]
        fc_layer_state_dict=[]
        for i,layer in model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_layer_state_dict.append([i, layer])
            if isinstance(layer, torch.nn.Conv2d):
                conv_layer_state_dict.append([i, layer])
            if isinstance(layer, torch.nn.Linear):
                fc_layer_state_dict.append([i, layer])
        # print(bn_layer_state_dict)
        # print(conv_layer_state_dict)
        convid = 0
        bnid = 0
        fcid = 0
        for name, layer in self.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                print('init layer '+name+' as ' + bn_layer_state_dict[bnid][0])
                layer.load_state_dict(bn_layer_state_dict[bnid][1].state_dict())
                bnid += 1
            if isinstance(layer, torch.nn.Linear):
                print('init layer '+name+' as ' + fc_layer_state_dict[fcid][0])
                layer.load_state_dict(fc_layer_state_dict[fcid][1].state_dict())
                fcid += 1
            if isinstance(layer, torch.nn.Conv2d):
                print('init layer '+name+' as ' + conv_layer_state_dict[convid][0])
                layer.weight=conv_layer_state_dict[convid][1].weight
                #             layer.
                #             layer.load_state_dict(conv_layer_state_dict[convid][1].state_dict())
                convid += 1
            # for i in layer.state_dict():
            #     print(i, layer.state_dict()[i].device)

    def load(self, args, strict=True):
        if args.pretrain:
            self.load_state_dict(torch.load(args.pretrain), strict=strict)