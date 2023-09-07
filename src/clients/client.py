import os
import torch
import random
import numpy as np

from torch.utils import data
from torch.cuda.amp import GradScaler
from torch import nn, distributed
from utils import HardNegativeMining, MeanReduction
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import SubsetRandomSampler
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class PairsSampler(DistributedSampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_pairs = len(dataset) // 2

    def __iter__(self):
        indices = list(range(self.num_pairs))
        random.shuffle(indices)
        # Espandi gli indici delle coppie in indici di immagini
        indices = [i * 2 for i in indices] + [i * 2 + 1 for i in indices]
        indices.sort()  # Ordina gli indici per mantenere la corrispondenza tra le coppie

        return iter(indices)

    def __len__(self):
        return len(self.dataset)


class Client:

    def __init__(self, args, client_id, dataset, model, writer, batch_size, world_size, rank, num_gpu,
                 device=None, test_user=False):

        self.args = args
        self.id = client_id
        self.dataset = dataset
        self.model = model
        self.writer = writer
        self.batch_size = batch_size
        self.device = device
        self.test_user = test_user
        self.num_gpu = num_gpu
        self.world_size = world_size
        self.rank = rank

        if self.args.mm_setting == "first" or self.args.mm_setting == "second":
            if self.dataset.root == 'data':
                self.format_client = "RGB"
                self.dataset.format_client = "RGB"

            if self.dataset.root == 'data/HHA_DATA':
                self.format_client = "HHA"
                self.dataset.format_client = "HHA"

        if self.args.mm_setting == 'third':
            self.format_client = "MIX"
            self.dataset.format_client = "MIX"




        if args.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(args.random_seed)
            if self.args.mm_setting != "third":
                self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, worker_init_fn=self.seed_worker,
                                              sampler=DistributedSampler(self.dataset, num_replicas=world_size, rank=rank),
                                              num_workers=4 * num_gpu, drop_last=True, pin_memory=True, generator=g)
                self.loader_full = data.DataLoader(self.dataset, batch_size=1, worker_init_fn=self.seed_worker,
                                                   sampler=DistributedSampler(self.dataset, num_replicas=world_size,
                                                                              rank=rank, shuffle=False),
                                                   num_workers=4 * num_gpu, drop_last=False, pin_memory=True, generator=g)
            else:

                # Creazione del campionatore personalizzato per le coppie
                pair_sampler = PairsSampler(self.dataset)

                self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, worker_init_fn=self.seed_worker,
                                              sampler=pair_sampler, num_workers=4 * num_gpu, drop_last=True, pin_memory=True, generator=g)
                self.loader_full = data.DataLoader(self.dataset, batch_size=1, worker_init_fn=self.seed_worker,
                                                   sampler=pair_sampler, num_workers=4 * num_gpu, drop_last=False, pin_memory=True, generator=g)
        else:
            self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
                                          sampler=DistributedSampler(self.dataset, num_replicas=world_size, rank=rank),
                                          num_workers=4 * num_gpu, drop_last=True, pin_memory=True)
            self.loader_full = data.DataLoader(self.dataset, batch_size=1,
                                               sampler=DistributedSampler(self.dataset, num_replicas=world_size,
                                                                          rank=rank, shuffle=False),
                                               num_workers=4 * num_gpu, drop_last=False, pin_memory=True)

        self.criterion, self.reduction = self.__get_criterion_and_reduction_rules()
        if self.args.mixed_precision:
            self.scaler = GradScaler()


        self.profiler_path = os.path.join('profiler', self.args.profiler_folder) if self.args.profiler_folder else None


    @staticmethod
    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def __get_criterion_and_reduction_rules(self):
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        return criterion, reduction

    @staticmethod
    def update_metric(metric, outputs, labels, **kwargs):
        _, prediction = outputs.max(dim=1)
        if prediction.shape != labels.shape:
            prediction = nn.functional.interpolate(
                prediction.unsqueeze(0).double(), labels.shape[1:], mode='nearest').squeeze(0).long()
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def L2Loss_function(self, outputs, labels):
        output_softmax = F.softmax(outputs, dim=1)
        mask_255 = labels == 255
        labels[mask_255] = 20
        one_hot_labels = F.one_hot(labels, 21).permute(0, 3, 1, 2)
        one_hot_labels = one_hot_labels[:, 0:19, :, :]
        loss_per_pixel = torch.norm(output_softmax - one_hot_labels, p=2, dim=1)

        return self.reduction(loss_per_pixel, labels)


    def aux_loss(self, f_RGB,f_HHA):
        loss_per_batch_element = torch.norm(f_RGB["out"] - f_HHA["out"], p=2, dim=(1,2,3))
        loss_per_batch_element = loss_per_batch_element/f_RGB.size(1)
        return loss_per_batch_element


    def calc_loss_and_output(self, images, labels):

        if self.args.model in ('deeplabv3',):
            outputs = self.model(images)['out']

            if self.args.Loss_funct_SS == "L2":
                loss_tot = self.L2Loss_function(outputs, labels)
            else:
                loss_tot = self.reduction(self.criterion(outputs, labels), labels)

            dict_calc_losses = {'loss_tot': loss_tot}
            return dict_calc_losses, outputs

        elif self.args.model in ('multi_deeplabv3',):
            if self.args.mm_setting=="second":
                if self.format_client == "RGB":

                    outputs=self.model.module.rgb_backbone(images)
                    outputs=self.model.module.classifier(outputs["out"])
                    outputs = F.interpolate(outputs, size=images.shape[-2:], mode='bilinear', align_corners=False)

                    if self.args.Loss_funct_SS == "L2":
                        loss_tot = self.L2Loss_function(outputs, labels)
                    else:
                        loss_tot = self.reduction(self.criterion(outputs, labels), labels)

                    dict_calc_losses = {'loss_tot': loss_tot}
                    return dict_calc_losses, outputs
                else:
                    outputs=self.model.module.hha_backbone(images)
                    outputs=self.model.module.classifier(outputs["out"])
                    outputs = F.interpolate(outputs, size=images.shape[-2:], mode='bilinear', align_corners=False)

                    if self.args.Loss_funct_SS == "L2":
                        loss_tot = self.L2Loss_function(outputs, labels)
                    else:
                        loss_tot = self.reduction(self.criterion(outputs, labels), labels)

                    dict_calc_losses = {'loss_tot': loss_tot}
                    return dict_calc_losses, outputs
            else:

                images = images.view(images.size(0) // 2, 2, images.size(1), images.size(2), images.size(3))
                x_rgb = images[:, 0, :, :]
                z_hha = images[:, 1, :, :]

                if self.args.Loss_funct_SS == "L2":
                    _, _, outputs = self.model(x_rgb=x_rgb, z_hha=z_hha)
                    loss_tot = self.L2Loss_function(outputs, labels)

                elif self.args.Loss_funct_SS == "CrossEnt":
                    _, _, outputs = self.model(x_rgb=x_rgb, z_hha=z_hha)
                    loss_tot = self.reduction(self.criterion(outputs, labels), labels)

                else:
                    l=0.1
                    f_RGB,f_HHA,outputs = self.model(x_rgb=x_rgb, z_hha=z_hha)
                    First_loss = self.criterion(outputs, labels)
                    Secnd_loss = self.aux_loss(f_RGB,f_HHA)

                    final_loss = First_loss + l*Secnd_loss

                    loss_tot = self.reduction(final_loss, labels)

                dict_calc_losses = {'loss_tot': loss_tot}

                return dict_calc_losses, outputs
        else:
            raise NotImplementedError

    def get_test_output(self, images):
        if self.args.model == 'deeplabv3':
            if self.args.fw_task == 'mcd':
                return self.model(images, classifier1=True)
            return self.model(images)['out']

        elif self.args.model == 'multi_deeplabv3':
            if self.args.mm_setting=="second":
                if self.format_client == "RGB":
                    outputs = self.model.module.rgb_backbone(images)
                    outputs = self.model.module.classifier(outputs['out'])
                else:
                    outputs = self.model.module.hha_backbone(images)
                    outputs = self.model.module.classifier(outputs['out'])
                return outputs
            else:
                batch_size = images.size(0)
                num_channels = images.size(1)
                height = images.size(2)
                width = images.size(3)
                images = images.view(batch_size // 2, 2, num_channels, height, width)
                x_rgb = images[:, 0, :, :]
                z_hha = images[:, 1, :, :]
                outputs = self.model(x_rgb=x_rgb, z_hha=z_hha)

                return outputs
        else:
            raise NotImplementedError

    def calc_test_loss(self, outputs, labels):

        if self.args.Loss_funct_SS == "L2":
            loss_tot = self.L2Loss_function(outputs, labels)

        else:
            loss_tot = self.reduction(self.criterion(outputs, labels), labels)

        return loss_tot

    def manage_tot_test_loss(self, tot_loss):
        tot_loss = torch.tensor(tot_loss).to(self.device)
        distributed.reduce(tot_loss, dst=0)
        return tot_loss / distributed.get_world_size() / len(self.loader)

    def run_epoch(self, cur_epoch, metrics, optimizer, scheduler=None):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        if self.args.mm_setting != "zero":
            return self.id+" "+self.format_client
        else:
            return self.id
    @property
    def num_samples(self):
        return len(self.dataset)

    @property
    def len_loader(self):
        return len(self.loader)
