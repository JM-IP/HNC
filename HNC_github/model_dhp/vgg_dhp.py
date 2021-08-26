import os

import torch
import torch.nn as nn
from model_dhp.dhp_base import DHP_Base, conv_dhp
import torch.utils.model_zoo as model_zoo
from model import common
from torch.autograd import Variable
import model_dhp.parametric_quantization as PQ


def make_model(args, cfg):
    if cfg.w_quantize is None:
        return VGG_DHP(args, cfg)
    else:
        return VGG_DHP(args, cfg, conv3x3=conv_dhp)


# reference: torchvision
class VGG_DHP(DHP_Base):
    def __init__(self, args, cfg, conv3x3=conv_dhp):
        super(VGG_DHP, self).__init__(args)
        # we use batch noramlization for VGG_DHP
        norm = None
        self.norm = norm
        bias = not args.no_bias
        self.cfg = cfg
        self.args = args
        self.embedding_dim = args.embedding_dim
        configs = {
            '7': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
            'ef': [32, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
        }

        body_list = []
        in_channels = args.n_colors
        self.latent_vector = nn.Parameter(torch.randn((in_channels)))
        assert args.data_train.find('CIFAR') >= 0 and args.vgg_type == '7'
        if args.data_train.find('CIFAR') >= 0 or args.data_train.find('Tiny') >= 0:

            for i, v in enumerate(configs[args.vgg_type]):
                if v == 'M':
                    body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    stride = 2 if i == 0 and args.data_train.find('Tiny') >= 0 else 1
                    if i == 7:
                        self.latent_vector_out = nn.Parameter(torch.randn((512)))
                        body_list.append(
                            conv_dhp(in_channels, v, kernel_size=args.kernel_size, stride=stride,
                                     bias=bias, latent_vector=self.latent_vector_out,
                                     embedding_dim=self.embedding_dim, cfg=self.cfg, args=self.args))
                    else:
                        body_list.append(conv_dhp(in_channels, v, kernel_size=args.kernel_size, stride=stride, bias=bias,
                                              embedding_dim=self.embedding_dim, cfg=self.cfg, args=self.args))
                    in_channels = v
        else:
            for i, v in enumerate(configs[args.vgg_type]):
                if v == 'M':
                    body_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    conv2d = conv3x3(in_channels, v, kernel_size=3)
                    if norm is not None:
                        body_list += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        body_list += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v

        self.features = nn.Sequential(*body_list)
        if args.data_train.find('CIFAR') >= 0:
            n_classes = int(args.data_train[5:])
            if args.template.find('linear3') >= 0:
                self.classifier = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Linear(in_channels, in_channels),
                                                nn.Linear(in_channels, n_classes))
            else:
                self.classifier = PQ.linear_Q_fn(self.cfg)(in_channels*4*4, n_classes)
        if args.data_train.find('Tiny') >= 0:
            n_classes = 200
            self.classifier = nn.Sequential(nn.Linear(in_channels, in_channels), nn.Linear(in_channels, in_channels),
                                            nn.Linear(in_channels, n_classes))
        elif args.data_train == 'ImageNet':
            n_classes = 1000
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, n_classes),
            )
            # self.classifier = nn.Sequential(
            #     nn.Linear(512 * 7 * 7, n_classes)
            # )
        # print(conv3x3, conv3x3 == common.default_conv or conv3x3 == nn.Conv2d)

        # if conv3x3 == common.default_conv or conv3x3 == nn.Conv2d:
        #     self.load(args, strict=True)
    def mask(self):
        masks = []
        for i, v in enumerate(self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)):
            channels = self.remain_channels(v)
            if i == 0 or i == 1:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))
            else:
                masks.append(v.abs() >= min(self.args.prune_threshold, v.abs().topk(channels)[0][-1]))
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        for i, v in enumerate(self.gather_latent_vector()):
            channels = self.remain_channels(v)
            if i != 0 and i != 1:
                if torch.sum(v.abs() >= self.pt) > channels:
                    if self.args.sparsity_regularizer == 'l1':
                        self.soft_thresholding(v, regularization)
                    elif self.args.sparsity_regularizer == 'l2':
                        self.proximal_euclidean_norm(v, regularization)
                    else:
                        raise NotImplementedError('Solution to regularization type {} is not implemented.'
                                                  .format(self.args.sparsity_regularizer))
    def set_parameters(self, calc_weight=False):
        latent_vectors, masks = self.gather_latent_vector(), self.mask()
        offset = 1
        for i, layer in enumerate(self.features):
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, conv_dhp)):
                if i == 0:
                    vm = [latent_vectors[0]] + [masks[0]] + [masks[2]]
                elif i==7:
                    vm = [latent_vectors[offset]] + [masks[offset]] + [masks[1]]
                else:
                    mask = masks[offset:offset+2]
                    vm = [latent_vectors[offset]] + mask
                offset += 1
                # print("DEBUG: ", vm)
                layer.set_parameters(vm, calc_weight)

    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            k = 1
            for i, layer in enumerate(self.features):
                if isinstance(layer, conv_dhp):
                    if i == 0:
                        x = layer([x, latent_vectors[0]])
                    else:
                        x = layer([x, latent_vectors[k]])
                    k = k + 1
                else:
                    x = layer(x)
        else:
            x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x

    def load(self, args, strict=True):
        model_dir = os.path.join('..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        if args.data_train.find('CIFAR') >= 0:
            if args.pretrain == 'download' or args.extend == 'download':
                url = (
                    'https://cv.snu.ac.kr/'
                    'research/clustering_kernels/models/vgg16-89711a85.pt'
                )

                state = model_zoo.load_url(url, model_dir=model_dir)
            elif args.pretrain:
                state = torch.load(args.pretrain)
            elif args.extend:
                state = torch.load(args.extend)
            else:
                common.init_vgg(self)
                return
        elif args.data_train == 'ImageNet':
            if args.pretrain == 'download':
                #print('pretrain download')
                if self.norm is not None:
                    url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
                else:
                    url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
                state = model_zoo.load_url(url, model_dir=model_dir)
            else:
                common.init_vgg(self)
                return
        elif args.data_train == 'Tiny_ImageNet':
            common.init_vgg(self)
            return
        else:
            raise NotImplementedError('Unavailable dataset {}'.format(args.data_train))
        #print(state['features.0.bias'])
        self.load_state_dict(state, strict=strict)
