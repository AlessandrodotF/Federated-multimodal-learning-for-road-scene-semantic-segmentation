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

        #if self.dataset.root == 'data/MIX_DATA':
        #    self.format_client = "MIX"
        #    self.dataset.format_client = "MIX"
        #    print("self.dataset.root ", self.dataset.root)



        if args.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(args.random_seed)
            self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, worker_init_fn=self.seed_worker,
                                          sampler=DistributedSampler(self.dataset, num_replicas=world_size, rank=rank),
                                          num_workers=4 * num_gpu, drop_last=True, pin_memory=True, generator=g)
            self.loader_full = data.DataLoader(self.dataset, batch_size=1, worker_init_fn=self.seed_worker,
                                               sampler=DistributedSampler(self.dataset, num_replicas=world_size,
                                                                          rank=rank, shuffle=False),
                                               num_workers=4 * num_gpu, drop_last=False, pin_memory=True, generator=g)
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

    def calc_loss_and_output(self, images, labels):

        if self.args.model in ('deeplabv3',):
            outputs = self.model(images)['out']
            loss_tot = self.reduction(self.criterion(outputs, labels), labels)
            dict_calc_losses = {'loss_tot': loss_tot}
            return dict_calc_losses, outputs

        elif self.args.model in ('multi_deeplabv3',):
            if self.format_client == "RGB":
                encoder = self.model.module.rgb_backbone
                outputs_encoder = encoder(images)
                outputs = self.model.module.classifier(outputs_encoder['out'])
            else:
                encoder = self.model.module.hha_backbone
                outputs_encoder = encoder(images)
                outputs = self.model.module.classifier(outputs_encoder['out'])
            if outputs.size() != labels.size():
                    outputs = F.interpolate(outputs, size=labels.size()[1:], mode='bilinear', align_corners=False)

            loss_tot = self.reduction(self.criterion(outputs, labels), labels)
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
            if self.format_client == "RGB":
                encoder = self.model.module.rgb_backbone
                outputs_encoder = encoder(images)
                outputs = self.model.module.classifier(outputs_encoder['out'])
            else:
                encoder = self.model.module.hha_backbone
                outputs_encoder = encoder(images)
                outputs = self.model.module.classifier(outputs_encoder['out'])
            return outputs

        else:
            raise NotImplementedError

    def calc_test_loss(self, outputs, labels):
        return self.reduction(self.criterion(outputs, labels), labels)

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
        return self.id+" "+self.format_client

    @property
    def num_samples(self):
        return len(self.dataset)

    @property
    def len_loader(self):
        return len(self.loader)
