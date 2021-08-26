"""
Designed three different forward passes.
1. thorough slicing: 1) slicing the weights and biaes in the hypernetwork. 2) for Batchnorm, use nn.ModuleList to
                        include batchnorms with different features.
           Problems: 1) slicing during every iteration is time-consuming. 2) Batchnorm problem -> need to
                        train every batchnorm in the ModuleList. The channels are not perfectly matched.
             Status: not implemented
2. no slicing and no masking: This is the easiest implementation. During the optimization/searching stage, do not slice
                        the weights or biases of the hypernetworks. Masks are not applied either. Corresponding to
                        module 'resnet_dhp.py' and 'resnet_dhp_share.py'. Note that for 'resnet_dhp_share.py', the
                        latent vectors are shared by ResBlocks in the same stage.
             Status: developing and chosen
3. no slicing but with masking: Do not slice the weights or biases of the hypernetworks but masks are applied to them.
                        This forward pass corresponds to module 'resnet_dhp_mask.py'.
             Status: developed but decided not to use it.
"""
import torch
import torch.nn as nn
from model_dhp.dhp_base import DHP_Base, conv_dhp
from IPython import embed
import model_dhp.parametric_quantization as PQ
from torch.autograd import Variable

def make_model(args, cfg, parent=False):
    return ResNet_DHP_ShareL(args, cfg)

class ResBlock_dhp(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=conv_dhp,
                 downsample=None, latent_vector=None,
                 embedding_dim=8, cfg=None, args=None, qact=True):
        super(ResBlock_dhp, self).__init__()
        expansion = 1
        self.finetuning = False
        self.cfg = cfg
        self.stride = stride
        self.args = args
        self.layer1 = conv3x3(in_channels, planes, kernel_size, stride=stride,
                              embedding_dim=embedding_dim, cfg=self.cfg, args=self.args, init_weight=False)
        self.layer2 = conv3x3(planes, expansion * planes, kernel_size, act=False,
                              latent_vector=latent_vector,
                              embedding_dim=embedding_dim, cfg=self.cfg, args=self.args, init_weight=False)
        self.downsample = downsample

        if qact:
            self.act_out = PQ.activation_Q_fn(self.cfg)()
        else:
            self.act_out = nn.ReLU(inplace=True)
        # self.act_out = PQ.activation_Q_fn(cfg)()

    def set_parameters(self, vector_mask, calc_weight=False):

        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        if self.downsample is not None:
            self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:], calc_weight)
        else:
            self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:0:-1], calc_weight)
        if self.downsample is not None:
            self.downsample.set_parameters(vector_mask[:2] + [vector_mask[-1]], calc_weight)
        self.act_out.channels = self.layer2.out_channels_remain if hasattr(self.layer2, 'out_channels_remain') \
            else self.layer2.out_channels

    def forward(self, x):
        if not self.finetuning:
            x, latent_input_vector = x
            # Need the latent vector of the previous layer.
            out = self.layer1([x, latent_input_vector])
            out = self.layer2([out, self.layer1.latent_vector])
            if self.downsample is not None:
                x = self.downsample([x, latent_input_vector])
        else:
            out = self.layer2(self.layer1(x))
            if self.downsample is not None:
                x = self.downsample(x)
        out += x
        out = self.act_out(out)
        return out
    def init_weight(self, latent_input_vector, conv_layer_state_dict, convid):
        # x, latent_input_vector = x
        print(convid)
        print( list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(self.layer1.bias2.data.shape).cuda().shape )
        print(self.layer1.bias2.data.shape)
        print(self.layer1.calc_weight(latent_input_vector).transpose(0, 1).data.shape)

        self.layer1.bias2.data = list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(self.layer1.bias2.data.shape).cuda() \
                            - (self.layer1.calc_weight(latent_input_vector).transpose(0, 1).data.reshape(self.layer1.bias2.data.shape))
        convid += 1
        print(list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(self.layer2.bias2.data.shape).cuda().shape)
        print(self.layer2.bias2.data.shape)
        print(self.layer2.calc_weight(self.layer1.latent_vector).transpose(0, 1).data.shape)

        self.layer2.bias2.data = list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(self.layer2.bias2.data.shape).cuda() \
                            - (self.layer2.calc_weight(self.layer1.latent_vector).transpose(0, 1).data.reshape(self.layer2.bias2.data.shape))
        convid += 1
        if self.downsample is not None:
            print(list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(self.downsample.bias2.data.shape).cuda().shape)
            print(self.downsample.bias2.data.shape)
            print(self.downsample.calc_weight(latent_input_vector).transpose(0, 1).data.shape)

            self.downsample.bias2.data = list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(self.downsample.bias2.data.shape).cuda() \
                                    - (self.downsample.calc_weight(latent_input_vector).transpose(0, 1).data.reshape(self.downsample.bias2.data.shape))
            convid += 1
        return convid


class BottleNeck_dhp(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, stride=1, conv3x3=conv_dhp,
                 downsample=None, latent_vector=None, embedding_dim=8, cfg=None, args=None):
        super(BottleNeck_dhp, self).__init__()
        expansion = 4
        self.finetuning = False
        self.stride = stride
        self.cfg = cfg
        self.layer1 = conv3x3(in_channels, planes, 1, embedding_dim=embedding_dim, cfg=cfg, init_weight=False)
        self.layer2 = conv3x3(planes, planes, kernel_size, stride=stride, embedding_dim=embedding_dim, cfg=cfg, init_weight=False)
        self.layer3 = conv3x3(planes, expansion*planes, 1, act=False, latent_vector=latent_vector, embedding_dim=embedding_dim, cfg=cfg, init_weight=False)
        self.downsample = downsample

        self.act_out = PQ.activation_Q_fn(cfg)()

    def set_parameters(self, vector_mask, calc_weight=False):
        self.layer1.set_parameters(vector_mask[:3], calc_weight)
        self.layer2.set_parameters([self.layer1.latent_vector] + vector_mask[2:4], calc_weight)
        if self.downsample is not None:
            self.layer3.set_parameters([self.layer2.latent_vector] + vector_mask[3:], calc_weight)
        else:
            self.layer3.set_parameters([self.layer2.latent_vector] + vector_mask[3:0:-2], calc_weight) #TODO: make sure this is correct
        if self.downsample is not None:
            self.downsample.set_parameters(vector_mask[:2] + [vector_mask[-1]], calc_weight)
        self.act_out.channels = self.layer3.out_channels_remain if hasattr(self.layer3, 'out_channels_remain') \
            else self.layer3.out_channels

    def forward(self, x):
        if not self.finetuning:
            x, latent_input_vector = x
            # Need the latent vector of the previous layer.
            out = self.layer1([x, latent_input_vector])
            out = self.layer2([out, self.layer1.latent_vector])
            out = self.layer3([out, self.layer2.latent_vector])
            if self.downsample is not None:
                x = self.downsample([x, latent_input_vector])
        else:
            out = self.layer3(self.layer2(self.layer1(x)))
            if self.downsample is not None:
                x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out

class ResNet_DHP_ShareL(DHP_Base):
    def __init__(self, args, cfg=None):
        super(ResNet_DHP_ShareL, self).__init__(args=args)
        self.width_mult = args.width_mult
        self.cfg = cfg
        self.args = args
        if args.depth == 18:
            self.expansion = 1
            self.block = ResBlock_dhp
            self.n_blocks = [2, 2, 2, 2]
        else:
            raise()
            # self.expansion = 4
            # self.block = BottleNeck_dhp
            # self.n_blocks = (args.depth - 2) // 9
        self.in_channels = int(64 * self.width_mult)
        self.latent_vector_stage0 = nn.Parameter(torch.randn(3))
        self.latent_vector_stage1 = nn.Parameter(torch.randn((int(64 * self.width_mult) * self.expansion)))
        self.latent_vector_stage2 = nn.Parameter(torch.randn((int(128 * self.width_mult) * self.expansion)))
        self.latent_vector_stage3 = nn.Parameter(torch.randn((int(256 * self.width_mult) * self.expansion)))
        self.latent_vector_stage4 = nn.Parameter(torch.randn((int(512 * self.width_mult) * self.expansion)))

        # self.features = nn.ModuleList([])
        self.features = nn.ModuleList([conv_dhp(args.n_colors, int(64 * self.width_mult),
                                                kernel_size=7,
                                                stride=2, latent_vector=self.latent_vector_stage1,
                                                embedding_dim=self.embedding_dim, cfg=self.cfg, args=self.args)])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.features.extend(self.make_layer(self.n_blocks[0], int(64 * self.width_mult), 3,
                                             stride=1, latent_vector=self.latent_vector_stage1))
        self.features.extend(self.make_layer(self.n_blocks[1], int(128 * self.width_mult), 3,
                                             stride=2, latent_vector=self.latent_vector_stage2))
        self.features.extend(self.make_layer(self.n_blocks[2], int(256 * self.width_mult), 3,
                                             stride=2, latent_vector=self.latent_vector_stage3))
        self.features.extend(self.make_layer(self.n_blocks[3], int(512 * self.width_mult), 3,
                                             stride=2, latent_vector=self.latent_vector_stage4, qact=False))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(int(512 * self.width_mult) * self.expansion, self.n_classes)
        # self.classifier = PQ.linear_Q_fn(self.cfg)(int(512 * self.width_mult), self.n_classes)
        self.show_latent_vector()

    def make_layer(self, blocks, planes, kernel_size, stride=1, conv3x3=conv_dhp,
                   latent_vector=None, qact=True):
        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = conv3x3(self.in_channels, out_channels, 1, stride=stride, act=False,
                                 latent_vector=latent_vector,
                                 embedding_dim=self.embedding_dim,
                                 cfg=self.cfg, args=self.args, init_weight=False)
        else:
            downsample = None
        kwargs = {'conv3x3': conv3x3, 'embedding_dim': self.embedding_dim}
        m = [self.block(self.in_channels, planes, kernel_size,
                        stride=stride, downsample=downsample,
                        latent_vector=latent_vector,
                        cfg=self.cfg, args=self.args, **kwargs)]
        self.in_channels = out_channels
        for i in range(blocks - 1):
            if i == blocks - 2:
                m.append(self.block(self.in_channels, planes, kernel_size,
                                    latent_vector=latent_vector, qact=qact,
                                    cfg=self.cfg, args=self.args, **kwargs))
            else:
                m.append(self.block(self.in_channels, planes, kernel_size,
                                    latent_vector=latent_vector, qact=True,
                                    cfg=self.cfg, args=self.args, **kwargs))

        return m

    def mask(self):#, get_channel_number=False
        masks = []
        for i, v in enumerate(self.gather_latent_vector(grad_prune=self.grad_prune, grad_normalize=self.grad_normalize)):
            channels = self.remain_channels(v)
            if i == 0 or i == 4:
                masks.append(torch.ones_like(v, dtype=torch.bool, device='cuda'))
            else:
                masks.append(v.abs() >= min(self.args.prune_threshold, v.abs().topk(channels)[0][-1]))
        return masks

    def proximal_operator(self, lr):
        regularization = self.regularization * lr
        for i, v in enumerate(self.gather_latent_vector()):
            channels = self.remain_channels(v)
            if i != 0 and i != 4:
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
        # former_vectors, last_vectors, other_vectors, former_masks, last_masks, other_masks = self.mask()
        for i, layer in enumerate(self.features):
            if i == 0:
                vm = [latent_vectors[0]] + masks[:2]
            else:
                mask = [masks[i + 4]]
                stage = (i - 1) // 2  # stage - 1
                if (i - 1) % 2 == 0 and i > 1:
                    vm = [latent_vectors[stage], masks[stage]] + mask + [masks[stage + 1]]
                else:
                    vm = [latent_vectors[stage + 1], masks[stage + 1]] + mask
            layer.set_parameters(vm, calc_weight)

    # def my_load(self, model):
    #     bn_layer_state_dict=[]
    #     conv_layer_state_dict=[]
    #     fc_layer_state_dict=[]
    #     for i,layer in model.named_modules():
    #         if isinstance(layer, torch.nn.BatchNorm2d):
    #             bn_layer_state_dict.append([i, layer])
    #         if isinstance(layer, torch.nn.Conv2d):
    #             conv_layer_state_dict.append([i, layer])
    #         if isinstance(layer, torch.nn.Linear):
    #             fc_layer_state_dict.append([i, layer])
    #     # print(bn_layer_state_dict)
    #     # print(conv_layer_state_dict)
    #     convid = 0
    #     bnid = 0
    #     fcid = 0
    #     for name, params in self.named_parameters():
    #         #     print(name)
    #         if 'latent_vector_stage' in name or 'weight1' in name:
    #             print('init layer '+name+' as: 1')
    #             torch.nn.init.constant_(params, 1)
    #         if 'weight2' in name:
    #             print('init layer '+name+' as: 1/embedding_dim')
    #             embedding_dim = self.embedding_dim
    #             torch.nn.init.constant_(params, 1/embedding_dim)
    #         if 'bias0' in name or 'bias1' in name:
    #             print('init layer '+name+' as: 0')
    #             torch.nn.init.constant_(params, 0)
    #         if 'bias2' in name:
    #             print('init layer '+name+' as ' + conv_layer_state_dict[convid][0])
    #             # print(list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].shape)
    #             # print(list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).shape)
    #             # print(params.data.shape)
    #             params.data = -1+list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(params.data.shape).cuda()
    #             convid += 1
    #     for name, layer in self.named_modules():
    #         if isinstance(layer, torch.nn.BatchNorm2d):
    #             print('init layer '+name+' as ' + bn_layer_state_dict[bnid][0])
    #             layer.load_state_dict(bn_layer_state_dict[bnid][1].state_dict())
    #             bnid += 1
    #         if isinstance(layer, torch.nn.Linear):
    #             print('init layer '+name+' as ' + fc_layer_state_dict[fcid][0])
    #             layer.load_state_dict(fc_layer_state_dict[fcid][1].state_dict())
    #             fcid += 1

    def my_load(self, model):
        bn_layer_state_dict = []
        conv_layer_state_dict = []
        fc_layer_state_dict = []
        PQ_layer_state_dict = []

        for i, layer in model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_layer_state_dict.append([i, layer])
            if isinstance(layer, torch.nn.Conv2d):
                conv_layer_state_dict.append([i, layer])
            if isinstance(layer, torch.nn.Linear):
                fc_layer_state_dict.append([i, layer])
            if isinstance(layer, PQ.parametric_fixed_point_quantize_d_xmax):
                if '.a_quant.a_quant' not in i:
                    PQ_layer_state_dict.append([i, layer])

        # print(bn_layer_state_dict)
        # print(conv_layer_state_dict)
        convid = 0
        bnid = 0
        fcid = 0
        PQid = 0
        for name, params in self.named_parameters():
            #     print(name)
            if 'latent_vector' in name:
                print('init layer ' + name + ' as: 0.1')
                torch.nn.init.constant_(params, 0.1)
            elif 'weight1' in name:
                print('init layer ' + name + ' as: 1')
                torch.nn.init.constant_(params, 1)
            elif 'weight2' in name:
                print('init layer ' + name + ' as: 100/embedding_dim')
                embedding_dim = self.embedding_dim
                torch.nn.init.constant_(params, 100 / embedding_dim)
            if 'bias0' in name or 'bias1' in name:
                print('init layer ' + name + ' as: 0')
                torch.nn.init.constant_(params, 0)
            if 'bias2' in name:
                print('init layer ' + name + ' as ' + conv_layer_state_dict[convid][0])
                # print(list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].shape)
                # print(list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).shape)
                # print(params.data.shape)
                params.data = -1 + list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0,
                                                                                                               1).reshape(
                    params.data.shape).cuda()
                convid += 1
        for name, layer in self.named_modules():
            if isinstance(layer, PQ.parametric_fixed_point_quantize_d_xmax):
                print('init layer ' + name + ' as ' + PQ_layer_state_dict[PQid][0])
                layer.load_state_dict(PQ_layer_state_dict[PQid][1].state_dict())
                PQid += 1
            if isinstance(layer, torch.nn.BatchNorm2d):
                print('init layer ' + name + ' as ' + bn_layer_state_dict[bnid][0])
                layer.load_state_dict(bn_layer_state_dict[bnid][1].state_dict())
                bnid += 1
            if isinstance(layer, torch.nn.Linear):
                print('init layer ' + name + ' as ' + fc_layer_state_dict[fcid][0])
                layer.load_state_dict(fc_layer_state_dict[fcid][1].state_dict())
                fcid += 1


    def forward(self, x):
        if not self.finetuning:
            latent_vectors = self.gather_latent_vector()
            for i, layer in enumerate(self.features):
                if i == 0:
                    x = layer([x, latent_vectors[0]])
                    x = self.maxpool(x)
                elif i <= 3:
                    x = layer([x, latent_vectors[1]])
                elif i <= 5:
                    x = layer([x, latent_vectors[2]])
                elif i <= 7:
                    x = layer([x, latent_vectors[3]])
                else:
                    x = layer([x, latent_vectors[4]])
        else:
            for i, layer in enumerate(self.features):
                if i == 0:
                    x = layer(x)
                    x = self.maxpool(x)
                else:
                    x = layer(x)
        x = self.avgpool(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

    def init_weight(self, model):
        bn_layer_state_dict=[]
        conv_layer_state_dict=[]
        fc_layer_state_dict=[]
        PQ_layer_state_dict=[]

        for i,layer in model.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_layer_state_dict.append([i, layer])
            if isinstance(layer, torch.nn.Conv2d):
                conv_layer_state_dict.append([i, layer])
            if isinstance(layer, torch.nn.Linear):
                fc_layer_state_dict.append([i, layer])
            if isinstance(layer, PQ.parametric_fixed_point_quantize_d_xmax):
                if '.a_quant.a_quant' not in i:
                    PQ_layer_state_dict.append([i, layer])

        # print(bn_layer_state_dict)
        # print(conv_layer_state_dict)
        convid = 0
        bnid = 0
        fcid = 0
        PQid = 0
        for name, layer in self.named_modules():
            if isinstance(layer, PQ.parametric_fixed_point_quantize_d_xmax):
                print('init layer '+name+' as ' + PQ_layer_state_dict[PQid][0])
                layer.load_state_dict(PQ_layer_state_dict[PQid][1].state_dict())
                PQid += 1
            if isinstance(layer, torch.nn.BatchNorm2d):
                print('init layer '+name+' as ' + bn_layer_state_dict[bnid][0])
                layer.load_state_dict(bn_layer_state_dict[bnid][1].state_dict())
                bnid += 1
            if isinstance(layer, torch.nn.Linear):
                print('init layer '+name+' as ' + fc_layer_state_dict[fcid][0])
                layer.load_state_dict(fc_layer_state_dict[fcid][1].state_dict())
                fcid += 1

                latent_vectors = self.gather_latent_vector()
        for i, layer in enumerate(self.features):
            if i == 0:
                print(list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(layer.bias2.data.shape).cuda().shape)
                print(layer.bias2.data.shape)
                print(layer.calc_weight(latent_vectors[0]).transpose(0, 1).data.shape)
                layer.bias2.data = list(conv_layer_state_dict[convid][1].state_dict().items())[0][1].transpose(0, 1).reshape(layer.bias2.data.shape).cuda() - \
                                   (layer.calc_weight(latent_vectors[0]).transpose(0, 1).data.reshape(layer.bias2.data.shape))
                convid += 1
            elif i <= 3:
                convid = layer.init_weight(latent_vectors[1], conv_layer_state_dict, convid)
            elif i <= 5:
                convid = layer.init_weight(latent_vectors[2], conv_layer_state_dict, convid)
            elif i <= 7:
                convid = layer.init_weight(latent_vectors[3], conv_layer_state_dict, convid)
            else:
                convid = layer.init_weight(latent_vectors[4], conv_layer_state_dict, convid)
        for i in self.gather_latent_vector():
            print(i)