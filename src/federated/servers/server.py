import copy
import numpy as np
import random
from torch import optim


class Server:

    def __init__(self, args, model, model_rgb, writer, local_rank, lr, momentum, optimizer, source_dataset):
        #questi credo di poterli lasciare invariati
        self.args=args
        self.momentum = momentum
        self.lr = lr
        self.writer = writer
        self.selected_clients = []
        self.local_rank = local_rank

        self.source_dataset = source_dataset
        self.opt_string = optimizer #è None

        self.total_grad = 0
        self.total_grad_rgb = 0

        self.model = copy.deepcopy(model)
        self.model_rgb = copy.deepcopy(model_rgb)

        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.model_rgb_params_dict = copy.deepcopy(self.model_rgb.state_dict())

        self.updates = []
        self.updates_rgb = []

        self.optimizer = self.__get_optimizer()
        self.optimizer_rgb = self.__get_optimizer_rgb()

        self.swa_model = None
        self.swa_model_rgb = None


    def train_source(self, *args, **kwargs):
        raise NotImplementedError

    def train_clients(self, *args, **kwargs):
        raise NotImplementedError

    def update_model(self):
        raise NotImplementedError

    def __get_optimizer(self):
        if self.opt_string is None:
            self.writer.write("Running without server optimizer")
            return None

        if self.opt_string == 'SGD':
            return optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)

        if self.opt_string == 'FedAvgm':
            return optim.SGD(params=self.model.parameters(), lr=1, momentum=0.9)

        if self.opt_string == 'Adam':
            return optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=10 ** (-1))

        if self.opt_string == 'AdaGrad':
            return optim.Adagrad(params=self.model.parameters(), lr=self.lr, eps=10 ** (-2))

        raise NotImplementedError
    def __get_optimizer_rgb(self):
        if self.opt_string is None:
            self.writer.write("Running without server optimizer")
            return None

        if self.opt_string == 'SGD':
            return optim.SGD(params=self.model_rgb.parameters(), lr=self.lr, momentum=self.momentum)

        if self.opt_string == 'FedAvgm':
            return optim.SGD(params=self.model_rgb.parameters(), lr=1, momentum=0.9)

        if self.opt_string == 'Adam':
            return optim.Adam(params=self.model_rgb.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=10 ** (-1))

        if self.opt_string == 'AdaGrad':
            return optim.Adagrad(params=self.model_rgb.parameters(), lr=self.lr, eps=10 ** (-2))

        raise NotImplementedError
    def select_clients(self, my_round, possible_clients, num_clients):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)


    """queste funzioni qui sotto per il caso attuale credo non servano"""
    def get_clients_info(self, clients):
        if clients is None:
            clients = self.selected_clients
        num_samples = {c.id: c.num_samples for c in clients}
        return num_samples

    @staticmethod
    def num_parameters(params):
        return sum(p.numel() for p in params if p.requires_grad)

    def setup_swa_model(self, swa_ckpt=None):
        self.swa_model = copy.deepcopy(self.model)
        if swa_ckpt is not None:
            self.swa_model.load_state_dict(swa_ckpt)

    def update_swa_model(self, alpha):
        for param1, param2 in zip(self.swa_model.parameters(), self.model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha


    def setup_swa_model_rgb(self, swa_ckpt=None):
        self.swa_model_rgb = copy.deepcopy(self.model_rgb)
        if swa_ckpt is not None:
            self.swa_model_rgb.load_state_dict(swa_ckpt)

    def update_swa_model_rgb(self, alpha):
        for param1, param2 in zip(self.swa_model_rgb.parameters(), self.model_rgb.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha