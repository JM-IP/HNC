# This module is used for differentiable meta pruning via hypernetworks.
from util import utility
import torch
import torch.nn as nn
from tqdm import tqdm
from model.in_use.flops_counter import get_model_flops
from model_dhp.flops_counter_dhp import get_flops, get_flops_prune_only, get_parameters_prune_only, \
    get_parameters, set_output_dimension, get_flops_grad
from model_dhp.dhp_base import set_finetune_flag
from tensorboardX import SummaryWriter
import os
import matplotlib
from torch.autograd import Variable
import torch.nn.functional as F
import math
import model_dhp.parametric_quantization as PQ
matplotlib.use('Agg')


class Trainer():
    def __init__(self, args, cfg, loader, my_model, my_loss=None, ckp=None, writer=None, converging=False):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.cfg = cfg
        self.var_weight = args.var_weight
        self.reg_weight = args.reg_weight
        self.writer = writer
        if args.data_train.find('CIFAR') >= 0:
            self.input_dim = (3, 32, 32)
        elif args.data_train.find('Tiny_ImageNet') >= 0:
            self.input_dim = (3, 64, 64)
        else:
            self.input_dim = (3, 224, 224)
        # if converging:
        #     self.reset_after_searching()
        #
        set_output_dimension(self.model.get_model(), self.input_dim)

        self.flops = get_flops(self.model.get_model(), cfg, init_flops=True, pruned=False)

        self.flops_prune = self.flops # at initialization, no pruning is conducted.
        self.flops_compression_ratio = self.flops_prune / self.flops
        self.params = get_parameters(self.model.get_model(), cfg, init_params=True)
        self.params_prune = self.params
        self.params_compression_ratio = self.params_prune / self.params

        self.flops_prune_only = get_flops(self.model.get_model(), cfg, init_flops=True, pruned=False)
        self.flops_compression_ratio_prune_only = (self.flops_prune_only / self.flops)
        self.params_prune_only = self.params
        self.params_compression_ratio_prune_only = self.params_prune_only / self.params
        self.flops_ratio_log = []
        self.params_ratio_log = []
        self.converging = converging
        self.ckp.write_log('\nThe computation complexity and number of parameters of the current network is as follows.'
                           '\nFlops: {:.4f} [G]\tParams {:.2f} [k]'.format(self.flops / 10. ** 9,
                                                                           self.params / 10. ** 3))
        # self.flops_another = get_model_flops(self.model.get_model(), self.input_dim, False)
        # self.ckp.write_log('Flops: {:.4f} [G] calculated by the original counter. \nMake sure that the two calculated '
        #                    'Flops are the same.\n'.format(self.flops_another / 10. ** 9))

        self.optimizer = utility.make_optimizer_dhp(args, self.model, ckp=ckp, converging=converging)
        self.scheduler = utility.make_scheduler_dhp(args, self.optimizer, args.decay.split('+')[0], converging=converging)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        if args.model.find('INQ') >= 0:
            self.inq_steps = args.inq_steps
        else:
            self.inq_steps = None

    def reset_after_searching(self):
        # Phase 1 & 3, model reset here.
        # PHase 2 & 4, model reset at initialization

        # In Phase 1 & 3, the optimizer and scheduler are reset.
        # In Phase 2, the optimizer and scheduler is not used.
        # In Phase 4, the optimizer and scheduler is already set during the initialization of the trainer.
        # during the converging stage, self.converging =True. Do not need to set lr_adjust_flag in make_optimizer_hinge
        #   and make_scheduler_hinge.
        if not self.converging and not self.args.test_only:
            self.model.get_model().reset_after_searching()
            self.converging = True
            del self.optimizer, self.scheduler
            torch.cuda.empty_cache()
            decay = self.args.decay if len(self.args.decay.split('+')) == 1 else self.args.decay.split('+')[1]
            self.optimizer = utility.make_optimizer_dhp(self.args, self.model, converging=self.converging)#, lr=
            self.scheduler = utility.make_scheduler_dhp(self.args, self.optimizer, decay,
                                                        converging=self.converging)


        self.flops_prune = get_flops(self.model.get_model(), self.cfg)
        self.flops_compression_ratio = self.flops_prune / self.flops
        self.params_prune = get_parameters(self.model.get_model(), self.cfg)
        self.params_compression_ratio = self.params_prune / self.params

        if not self.args.test_only and self.args.summary:
            self.writer = SummaryWriter(os.path.join(self.args.dir_save, self.args.save), comment='converging')
        if os.path.exists(os.path.join(self.ckp.dir, 'epochs.pt')):
            self.epochs_searching = torch.load(os.path.join(self.ckp.dir, 'epochs.pt'))

    def train(self):
        epoch, lr = self.start_epoch()
        self.model.begin(epoch, self.ckp)
        self.loss.start_log()
        timer_data, timer_model = utility.timer(), utility.timer()
        # timer_back = utility.timer()
        # timer_opt = utility.timer()
        # timer_l1 = utility.timer()
        # timer_setp = utility.timer()
        # timer_ford = utility.timer()
        n_samples = 0

        for batch, (img, label) in enumerate(self.loader_train):
            img, label = self.prepare(img, label)
            n_samples += img.size(0)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            # print('DEBUG: ============> model forward:')
            prediction = self.model(img)
            # print('DEBUG: ============> model forward end')
            loss, _ = self.loss(prediction, label)
            # timer_model.hold()
            reg_loss = Variable(torch.Tensor([-1]).cuda(), requires_grad=False)
            var_loss = Variable(torch.Tensor([-1]).cuda(), requires_grad=False)

            # if self.var_weight>0:
            #     var_loss = 0
            #     for latent in self.model.get_model().gather_latent_vector()[2::]:
            #         x = F.normalize(latent.abs(), p=1, dim=0)
            #         var_loss -= torch.var(x)
            #  + self.var_weight * var_loss
            # timer_back.tic()

            # timer_back.hold()
            if not self.converging:
                # print('DEBUG: ============> model set parameters')
                # timer_l1.tic()
                self.model.get_model().proximal_operator(lr)
                # if batch % self.args.compression_check_frequency == 0:
                self.model.get_model().set_parameters()
                #     self.flops_prune = get_flops(self.model.get_model(), self.cfg)
                #     self.flops_compression_ratio = (self.flops_prune / self.flops)
                #     self.params_prune = get_parameters(self.model.get_model(), self.cfg)
                #     self.params_compression_ratio = self.params_prune / self.params
                #     self.flops_ratio_log.append(self.flops_compression_ratio)
                #     self.params_ratio_log.append(self.params_compression_ratio)
                #     self.flops_prune_only =  get_flops_prune_only(self.model.get_model(), self.cfg,pruned=False)
                #     self.flops_compression_ratio_prune_only = (self.flops_prune_only / self.flops)
                #     if self.flops_compression_ratio_prune_only>1:
                #         raise('why % > 1')
                #     self.params_prune_only = get_parameters_prune_only(self.model.get_model(), self.cfg)
                #     self.params_compression_ratio_prune_only = self.params_prune_only / self.params
                # if batch % 300 == 0:
                #     self.model.get_model().latent_vector_distribution(epoch, batch + 1, self.ckp.dir)
                #     self.model.get_model().per_layer_compression_ratio_quantize_precision(epoch, batch + 1, self.ckp.dir, cfg=self.cfg)

                # timer_l1.hold()
                # timer_setp.tic()
                #
                # timer_setp.hold()
                # print('DEBUG: ============> model set parameters end')'
                t = self.args.ratio * self.flops
                if self.args.gradually:
                    t = t * (10-epoch//10+1)/2
                    # t = t * (3-epoch//10+1)/2
                reg_loss =  torch.log(torch.max((get_flops_grad(self.model.get_model(), self.cfg)/t).cuda(),
                                      Variable(torch.Tensor([1]).cuda(), requires_grad=False)))
                (loss + self.reg_weight * reg_loss).backward()
                # # VAR_LOSS V1
                # for latent in self.model.get_model().gather_latent_vector()[1::]:
                #     var_loss += torch.var(latent)
                # # VAR_LOSS V2
                # for latent in self.model.get_model().gather_latent_vector()[1::]:
                #     var_loss -= torch.var(latent)
                # VAR_LOSS V4
                PQ.clip_quant_grads(self.model.get_model(), self.cfg)
                # timer_opt.tic()
                self.optimizer.step()
                # timer_opt.hold()
                # print('yjm is here to debug timeback is:', timer_model.release(), timer_back.release(), timer_opt.release(),
                #       timer_l1.release(), timer_setp.release())
                PQ.clip_quant_vals(self.model.get_model(), self.cfg)

            else:
                loss.backward()
                self.optimizer.step()



        # proximal operator
            if not self.converging:
                if batch % self.args.compression_check_frequency == 0 or batch==0:
                    self.flops_prune = get_flops(self.model.get_model(), self.cfg)
                    self.flops_compression_ratio = (self.flops_prune / self.flops)
                    self.params_prune = get_parameters(self.model.get_model(), self.cfg)
                    self.params_compression_ratio = self.params_prune / self.params
                    self.flops_ratio_log.append(self.flops_compression_ratio)
                    self.params_ratio_log.append(self.params_compression_ratio)
                    self.flops_prune_only =  get_flops_prune_only(self.model.get_model(), self.cfg,pruned=False)
                    self.flops_compression_ratio_prune_only = (self.flops_prune_only / self.flops)
                    self.params_prune_only = get_parameters_prune_only(self.model.get_model(), self.cfg)
                    self.params_compression_ratio_prune_only = self.params_prune_only / self.params
                if batch % 300 == 0 or batch==0:
                    for name,v in self.model.get_model().named_modules():
                        #     print(name)
                        if 'quantize' in name:
                            print("layer {}, \t\tw precision: {}, \tdelta: {}, \txmax: {}".format(
                                name, PQ.get_percision(v, self.cfg).item(), v.d.item(), v.xmax.item()))
                        if 'act_out.a_quant' in name or 'a_quant.a_quant' in name:
                            try:
                                print("layer {}, \tap precision: {}, \tdelta: {}, \txmax: {}".format(
                                name, PQ.get_percision_a(v, self.cfg).item(), v.d.item(), v.xmax.item()))
                            except IOError:
                                pass
                            else:
                                pass
                    self.model.get_model().latent_vector_distribution(epoch, batch + 1, self.ckp.dir)
                    self.model.get_model().per_layer_compression_ratio_quantize_precision(epoch, batch + 1, self.ckp.dir, cfg=self.cfg)


            timer_model.hold()

            if batch % self.args.print_every == 0 or batch==0:
                self.ckp.write_log('{}/{} ({:.0f}%)\t'
                                   'NLL: {:.3f}\tTop1: {:.2f} / Top5: {:.2f}\t'
                                   'Params Loss: {:.3f}\t'
                                   'Var Loss: {:.3f}\t'
                                   'Time: {:.1f}+{:.1f}s\t'
                                   'Flops Ratio: {:.2f}% = {:.4f} [G] / {:.4f} [G]\t'
                                   'Params Ratio: {:.2f}% = {:.2f} [k] / {:.2f} [k]\t'
                                   'Flops_prune_only Ratio: {:.2f}% = {:.4f} [G] / {:.4f} [G]\t'
                                   'Params_prune_only Ratio: {:.2f}% = {:.2f} [k] / {:.2f} [k]'.format(
                    n_samples, len(self.loader_train.dataset), 100.0 * n_samples / len(self.loader_train.dataset),
                    *(self.loss.log_train[-1, :] / n_samples), reg_loss.item(), var_loss.item(),
                    timer_model.release(), timer_data.release(),
                    self.flops_compression_ratio * 100, self.flops_prune / 10. ** 9, self.flops / 10. ** 9,
                    self.params_compression_ratio * 100, self.params_prune / 10. ** 3, self.params / 10. ** 3,
                    self.flops_compression_ratio_prune_only * 100, self.flops_prune_only / 10. ** 9, self.flops / 10. ** 9,
                    self.params_compression_ratio_prune_only * 100, self.params_prune_only / 10. ** 3, self.params / 10. ** 3))


            if self.args.summary:
                if (batch + 1) % 50 == 0:
                    self.writer.add_scalar('regloss/', reg_loss.item(),
                                           1000 * (epoch - 1) + batch)
                    for name, param in self.model.named_parameters():
                        if name.find('features') >= 0 and name.find('weight') >= 0:
                            self.writer.add_scalar('data/' + name, param.clone().cpu().data.abs().mean().numpy(),
                                                   1000 * (epoch - 1) + batch)
                            if param.grad is not None:
                                self.writer.add_scalar('data/' + name + '_grad',
                                                       param.grad.clone().cpu().data.abs().mean().numpy(),
                                                       1000 * (epoch - 1) + batch)
                if (batch + 1) == 500:
                    self.writer.add_scalar('regloss/', reg_loss.item(),
                                           1000 * (epoch - 1) + batch)
                    for name, param in self.model.named_parameters():
                        if name.find('features') >= 0 and name.find('weight') >= 0:
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), 1000 * (epoch - 1) + batch)
                            if param.grad is not None:
                                self.writer.add_histogram(name + '_grad', param.grad.clone().cpu().data.numpy(),
                                                      1000 * (epoch - 1) + batch)

            timer_data.tic()

        if not self.converging:
            self.model.get_model().set_parameters()
            self.flops_prune = get_flops(self.model.get_model(), self.cfg)
            self.flops_compression_ratio = (self.flops_prune / self.flops)
            self.params_prune = get_parameters(self.model.get_model(), self.cfg)
            self.params_compression_ratio = self.params_prune / self.params
            self.flops_ratio_log.append(self.flops_compression_ratio)
            self.params_ratio_log.append(self.params_compression_ratio)
            self.flops_prune_only =  get_flops_prune_only(self.model.get_model(), self.cfg,pruned=False)
            self.flops_compression_ratio_prune_only = (self.flops_prune_only / self.flops)
            self.params_prune_only = get_parameters_prune_only(self.model.get_model(), self.cfg)
            self.params_compression_ratio_prune_only = self.params_prune_only / self.params
            # if not self.converging and self.terminate():
            #     break
            self.ckp.write_log('{}/{} ({:.0f}%)\t'
                               'NLL: {:.3f}\tTop1: {:.2f} / Top5: {:.2f}\t'
                               'Params Loss: {:.3f}\t'
                               'Var Loss: {:.3f}\t'
                               'Flops Ratio: {:.2f}% = {:.4f} [G] / {:.4f} [G]\t'
                               'Params Ratio: {:.2f}% = {:.2f} [k] / {:.2f} [k]\t'
                               'Flops_prune_only Ratio: {:.2f}% = {:.4f} [G] / {:.4f} [G]\t'
                               'Params_prune_only Ratio: {:.2f}% = {:.2f} [k] / {:.2f} [k]'.format(
                n_samples, len(self.loader_train.dataset), 100.0 * n_samples / len(self.loader_train.dataset),
                *(self.loss.log_train[-1, :] / n_samples),
                reg_loss.item(), var_loss.item(), self.flops_compression_ratio * 100,
                self.flops_prune / 10. ** 9, self.flops / 10. ** 9,
                self.params_compression_ratio * 100,
                self.params_prune / 10. ** 3, self.params / 10. ** 3,
                self.flops_compression_ratio_prune_only * 100,
                self.flops_prune_only / 10. ** 9, self.flops / 10. ** 9,
                self.params_compression_ratio_prune_only * 100,
                self.params_prune_only / 10. ** 3, self.params / 10. ** 3))

            self.model.get_model().latent_vector_distribution(epoch, 1, self.ckp.dir)
            self.model.get_model().per_layer_compression_ratio_quantize_precision(epoch, 1, self.ckp.dir, cfg=self.cfg)


        self.model.log(self.ckp) # TODO: why this is used?
        self.loss.end_log(len(self.loader_train.dataset))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.loss.start_log(train=False)
        self.model.eval()

        timer_test = utility.timer()
        timer_test.tic()
        with torch.no_grad():
            for img, label in tqdm(self.loader_test, ncols=80):
                img, label = self.prepare(img, label)
                prediction = self.model(img)
                self.loss(prediction, label, train=False)

        self.loss.end_log(len(self.loader_test.dataset), train=False)

        # Lower is better
        best = self.loss.log_test.min(0)
        for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
            self.ckp.write_log('{}: {:.3f} (Best: {:.3f} from epoch {})'.
                               format(measure, self.loss.log_test[-1, i], best[0][i], best[1][i] + 1))

        if hasattr(self, 'epochs_searching') and self.converging:
            best = self.loss.log_test[:self.epochs_searching, :].min(0)
            self.ckp.write_log('\nBest during searching')
            for i, measure in enumerate(('Loss', 'Top1 error', 'Top5 error')):
                self.ckp.write_log('{}: {:.3f} from epoch {}'.format(measure, best[0][i], best[1][i]))
        self.ckp.write_log('Time: {:.2f}s\n'.format(timer_test.toc()), refresh=True)

        is_best = self.loss.log_test[-1, self.args.top] <= best[0][self.args.top]
        self.ckp.save(self, epoch, converging=self.converging, is_best=is_best)
        self.ckp.save_results(epoch, self.model)

        # scheduler.step is moved from training procedure to test procedure
        self.scheduler.step()

    def prepare(self, *args):
        def _prepare(x):
            x = x.to(self.device)
            if self.args.precision == 'half':
                x = x.half()
            return x

        return [_prepare(a) for a in args]

    def start_epoch(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        if not self.converging:
            stage = 'Searching Stage'
        else:
            stage = 'Converging Stage (Searching Epoch {})'.format(self.epochs_searching)
        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2}\t{}'.format(epoch, lr, stage))
        return epoch, lr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            if self.converging:
                return epoch >= self.args.epochs
            else:
                print("current ratio: ", self.flops_compression_ratio, " target ratio: ",self.args.ratio, "is end? : ",
                      ((self.flops_compression_ratio - self.args.ratio) <= self.args.stop_limit or epoch > 99))

                return ((self.flops_compression_ratio - self.args.ratio) <= self.args.stop_limit or epoch > 99) and epoch>=30
