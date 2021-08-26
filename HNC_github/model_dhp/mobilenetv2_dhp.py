'''
This module applies diffrentiable pruning via hypernetworks to MobileNetV2.

MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_dhp.dhp_base import DHP_Base, conv_dhp
from IPython import embed
import model_dhp.parametric_quantization as PQ

def make_model(args, cfg, parent=False):
    return MobileNetV2_DHP(args, cfg)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

def convert_index(i):
    if i == 0:
        x = 0
    elif i == 1:
        x = 8
    elif i == 2:
        x = 1
    elif 3 <= i < 5:
        x = 2
    elif 5 <= i < 8:
        x = 3
    elif 8 <= i < 12:
        x = 4
    elif 12 <= i < 15:
        x = 5
    elif 15 <= i < 18:
        x = 6
    else:
        x = 7
    return x


class InvertedResidual_dhp(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, latent_vector=None, embedding_dim=8, cfg=None, args=None):
        super(InvertedResidual_dhp, self).__init__()
        self.in_channels = inp
        self.out_channels = oup
        self.stride = stride
        self.cfg = cfg
        self.args = args
        self.finetuning = False
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.layers = nn.ModuleList()
        if expand_ratio != 1:
            # pw
            self.layers.append(conv_dhp(inp, hidden_dim, 1, embedding_dim=embedding_dim, cfg=self.cfg, args=self.args))
        self.layers.append(conv_dhp(hidden_dim, hidden_dim, 3, stride=stride, groups=hidden_dim, embedding_dim=embedding_dim, cfg=self.cfg, args=self.args))
        self.layers.append(conv_dhp(hidden_dim, oup, 1, act=True, latent_vector=latent_vector, embedding_dim=embedding_dim, cfg=self.cfg, args=self.args))

    def set_parameters(self, vector_mask, calc_weight=False):
        self.layers[0].set_parameters(vector_mask[:3], calc_weight)
        if len(self.layers) == 2:
            self.layers[1].set_parameters(vector_mask[:2] + [vector_mask[3]], calc_weight)
        else:
            self.layers[1].set_parameters([self.layers[0].latent_vector] + vector_mask[2:4], calc_weight)
            if self.in_channels == self.out_channels:
                m = [vector_mask[1]]
            else:
                m = [vector_mask[-1]]
            self.layers[2].set_parameters([self.layers[0].latent_vector] + [vector_mask[2]] + m, calc_weight)

    def forward(self, x):

        if not self.finetuning:
            x, latent_vector_input = x
            out = self.layers[0]([x, latent_vector_input])
            if len(self.layers) == 2:
                out = self.layers[1]([out, latent_vector_input])
            else:
                out = self.layers[1]([out, self.layers[0].latent_vector])
                out = self.layers[2]([out, self.layers[0].latent_vector])
        else:
            out = self.layers[0](x)
            for layer in self.layers[1:]:
                out = layer(out)

        if self.use_res_connect:
            out += x
        return out


# (expansion, out_planes, num_blocks, stride)
netcfg = [(1,  16, 1, 1),
       (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       (6,  96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]

netcfg_imagenet = [(1,  16, 1, 1),
       (6,  24, 2, 2),
       (6,  32, 3, 2),
       (6,  64, 4, 2),
       (6,  96, 3, 1),
       (6, 160, 3, 2),
       (6, 320, 1, 1)]


class MobileNetV2_DHP(DHP_Base):

    def __init__(self, args, cfg=None):
        super(MobileNetV2_DHP, self).__init__(args=args)
        self.width_mult = args.width_mult
        self.prune_classifier = args.prune_classifier
        self.cfg = cfg
        self.args = args
        if args.data_train == 'ImageNet':
            self.netcfg = netcfg_imagenet
        else:
            self.netcfg = netcfg

        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        stride = 1 if args.data_train.find('CIFAR') >= 0 else 2

        self.latent_vectors = nn.ParameterList([nn.Parameter(torch.randn((3)))] +
                                               [nn.Parameter(torch.randn((int(c[1] * self.width_mult)))) for c in self.netcfg])

        self.features = nn.ModuleList([conv_dhp(3, int(32 * self.width_mult), kernel_size=3, stride=stride, embedding_dim=self.embedding_dim, cfg=self.cfg, args=self.args)])
        self.features.extend(self._make_layers(in_planes=int(32 * self.width_mult)))
        if args.quan_last_conv:
            self.features.append(conv_dhp(int(320 * self.width_mult), int(1280 * self.width_mult), kernel_size=1, stride=1, embedding_dim=self.embedding_dim, cfg=self.cfg, args=self.args))
        else:
            self.features.append(ConvBNReLU(int(320 * self.width_mult), int(1280 * self.width_mult), kernel_size=1, stride=1))
        # self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(int(1280 * self.width_mult), self.n_classes))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(int(1280 * self.width_mult), self.n_classes))
        # nn.Sequential(PQ.linear_Q_fn(self.cfg)(int(1280 * self.width_mult), self.n_classes))
        self.show_latent_vector()

    def _make_layers(self, in_planes):
        layers = []
        for i, (expansion, out_planes, num_blocks, stride) in enumerate(self.netcfg):
            out_planes = int(out_planes * self.width_mult)
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(InvertedResidual_dhp(in_planes, out_planes, stride, expansion, latent_vector=self.latent_vectors[i + 1], embedding_dim=self.embedding_dim, cfg=self.cfg, args=self.args))
                in_planes = out_planes
        return layers

    def mask(self):
        masks = []
        for i, v in enumerate(self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)):
            # if (not self.prune_classifier and i != 0 and i != 42 and not (i > 8 and i % 2 == 1)) \
            #         and (self.prune_classifier and i != 0 and not (i > 8 and i % 2 == 1)):
                # embed()
            if not self.prune_classifier:
                flag = i != 0 and i != 42 and not (i > 8 and i % 2 == 1)
            else:
                flag = i != 0 and not (i > 8 and i % 2 == 1)
            if flag:
                if flag == 42:
                    channels = self.remain_channels(v, percentage=self.linear_percentage)
                else:
                    channels = self.remain_channels(v)
                masks.append(v.abs() >= min(self.pt, v.abs().topk(channels)[0][-1]))
            else:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))
        # mask has no gradients. Comparison operations are without gradients automatically.
        # when calculating the weights, biases, don't need to slice them?
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        for i, v in enumerate(self.gather_latent_vector()):
            if not self.prune_classifier:
                flag = i != 0 and i != 42 and not (i > 8 and i % 2 == 1)
            else:
                flag = i != 0 and not (i > 8 and i % 2 == 1)
            if flag:
                if flag == 42:
                    channels = self.remain_channels(v, percentage=self.linear_percentage)
                else:
                    channels = self.remain_channels(v)
                # if (not self.prune_classifier and i != 0 and i != 42 and not (i > 8 and i % 2 == 1):
                if torch.sum(v.abs() >= self.pt) > channels:
                    self.soft_thresholding(v, regularization)

    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()

        # former_vectors, last_vectors, other_vectors, former_masks, last_masks, other_masks = self.mask()
        for i, layer in enumerate(self.features):
            j = convert_index(i)
            if i == 0:
                vm = [latent_vectors[j], masks[j], masks[8]]
            elif i == 1:
                vm = [latent_vectors[j], masks[j], masks[9], masks[1]]
            elif i == 18:
                vm = [latent_vectors[j], masks[j], masks[-1]]
            elif i == 2 or i == 4 or i == 7 or i == 11 or i == 14 or i == 17:
                vm = [latent_vectors[j], masks[j]] + masks[2 * i + 6: 2 * i + 8] + [masks[j + 1]]
            else:
                vm = [latent_vectors[j], masks[j]] + masks[2 * i + 6: 2 * i + 8]
            layer.set_parameters(vm, calc_weight)
        if self.prune_classifier:
            super(MobileNetV2_DHP, self).set_parameters(calc_weight)
            mask = self.mask()[42]
            mask_input = mask.to(torch.float32).nonzero().squeeze(1)
            self.classifier[1].in_features_remain = mask_input.shape[0]
            if calc_weight:
                self.classifier[1].in_features = mask_input.shape[0]
                # embed()
                # if mask_input.shape[0] < 1280:
                #     mask_input = mask_input.cpu().numpy().tolist()
                #     num = len(mask_input)
                #     mask_other = sorted(list(set(range(mask.shape[0])) - set(mask_input)))
                #     index = sorted(random.sample(range(len(mask_other)), 1280 - num))
                #     for i in index:
                #         mask_input.append(mask_other[i])
                #     mask_input = torch.tensor(sorted(mask_input)).cuda()
                weight = self.classifier[1].weight.data
                weight = torch.index_select(weight, dim=1, index=mask_input)
                self.classifier[1].weight = nn.Parameter(weight)

    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            # for i, (k, v) in enumerate(zip(kk, latent_vectors)):
            #     print(i, list(v.shape), k)
            # embed()
            for i, layer in enumerate(self.features):
                j = convert_index(i)
                x = layer([x, latent_vectors[j]])
        else:
            for i, layer in enumerate(self.features):
                x = layer(x)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(x, 4)
        # out = out.view(out.size(0), -1)
        out = x.mean([2, 3])
        out = self.classifier(out)
        return out
