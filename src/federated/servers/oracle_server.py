import copy

import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from federated.servers.server import Server
from PIL import Image

class OracleServer(Server):

    def __init__(self, args, model, writer, local_rank, lr, momentum, optimizer=None, source_dataset=None, model_rgb=None):
        super().__init__(args, model, writer, local_rank, lr, momentum, optimizer=optimizer, source_dataset=source_dataset,  model_rgb=model_rgb)

    def train_source(self, *args, **kwargs):
        pass

    def _compute_client_delta(self, cmodel):
        delta = OrderedDict.fromkeys(cmodel.keys())
        for k, x, y in zip(self.model_params_dict.keys(), self.model_params_dict.values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def _compute_client_delta_rgb(self, cmodel):
        delta = OrderedDict.fromkeys(cmodel.keys())
        for k, x, y in zip(self.model_rgb_params_dict.keys(), self.model_rgb_params_dict.values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def train_clients(self, partial_metric=None, r=None, metrics=None, target_test_client=None, test_interval=None,
                      ret_score='Mean IoU', partial_metric_2=None):
        # self.optimizer = None
        if self.args.mm_setting=="first":
            losses = {}
            losses_rgb = {}
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            if self.optimizer_rgb is not None:
                self.optimizer_rgb.zero_grad()
        else:
            losses = {}
            if self.optimizer is not None:
                self.optimizer.zero_grad()


        clients = self.selected_clients

        for i, c in enumerate(clients):
            if self.args.mm_setting =="zero":
                self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c.id} ")
            else:
                self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c.id} {c.format_client}")

            # FINO QUI VENGONO PASSATI CORRETTAMENTE LE COPPIE DI IMMAGINI E LE COPPIE DI LABELS
            if self.args.mm_setting=="first":
                if c.format_client=="HHA":
                    c.model.load_state_dict(self.model_params_dict)
                    #def train in oracle client
                    out = c.train(partial_metric, r=r)
                    if self.local_rank == 0:
                        num_samples, update, dict_losses_list = out
                        losses[c.id+c.format_client] = {'loss': dict_losses_list, 'num_samples': num_samples}
                    else:
                        num_samples, update = out
                    #self.optimizer_rgb is NONE
                    if self.optimizer is not None:
                        update = self._compute_client_delta(update)
                    self.updates.append((num_samples, update))
                else:
                    c.model.load_state_dict(self.model_rgb_params_dict)
                    out = c.train(partial_metric_2, r=r)
                    if self.local_rank == 0:
                        num_samples, update, dict_losses_rgb_list = out
                        losses_rgb[c.id+c.format_client] = {'loss': dict_losses_rgb_list, 'num_samples': num_samples}
                    else:
                        num_samples, update = out
                    if self.optimizer_rgb is not None:
                        update = self._compute_client_delta_rgb(update)
                    self.updates_rgb.append((num_samples, update))

            elif self.args.mm_setting == "second":
                if c.format_client == "HHA":
                    c.model.module.hha_backbone.load_state_dict(self.hha_backbone_params_dict)
                    c.model.module.classifier.load_state_dict(self.classifier_params_dict)

                    # def train in oracle client
                    out = c.train(partial_metric, r=r)
                    if self.local_rank == 0:
                        num_samples, update, dict_losses_list = out
                        losses[c.id + c.format_client] = {'loss': dict_losses_list, 'num_samples': num_samples}
                    else:
                        num_samples, update = out
                    # self.optimizer_rgb is NONE
                    if self.optimizer is not None:
                        update = self._compute_client_delta(update)
                    self.updates.append((num_samples, update))
                    print("end")

                else:
                    c.model.module.rgb_backbone.load_state_dict(self.rgb_backbone_params_dict)
                    c.model.module.classifier.load_state_dict(self.classifier_params_dict)
                    out = c.train(partial_metric, r=r)
                    if self.local_rank == 0:
                        num_samples, update, dict_losses_list = out
                        losses[c.id + c.format_client] = {'loss': dict_losses_list, 'num_samples': num_samples}
                    else:
                        num_samples, update = out
                    if self.optimizer is not None:
                        update = self._compute_client_delta(update)
                    self.updates_rgb.append((num_samples, update))
                    print("end")
            else:
                c.model.load_state_dict(self.model_params_dict)
                out = c.train(partial_metric, r=r)
                if self.local_rank == 0:
                    num_samples, update, dict_losses_list = out
                    losses[c.id] = {'loss': dict_losses_list, 'num_samples': num_samples}
                else:
                    num_samples, update = out
                if self.optimizer is not None:
                    update = self._compute_client_delta(update)

                self.updates.append((num_samples, update))


        if self.local_rank == 0 and self.args.mm_setting=="first":
            return losses, losses_rgb
        if self.local_rank == 0 and self.args.mm_setting!="first":
            return losses

        return None

    def _aggregation(self):
        total_weight = 0.
        base = OrderedDict()
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to(self.local_rank) / total_weight
        return averaged_sol_n
    def _aggregation_rgb(self):
        total_weight = 0.
        base = OrderedDict()
        for (client_samples, client_model) in self.updates_rgb:
            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
        averaged_sol_n_rgb = copy.deepcopy(self.model_rgb_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n_rgb[key] = value.to(self.local_rank) / total_weight

        return averaged_sol_n_rgb

    def _aggregation_second_exp(self):
        total_weight_hha = 0.
        total_weight_rgb = 0.
        base_hha = OrderedDict()
        base_rgb = OrderedDict()
        base_decoder = OrderedDict()

        for (client_samples, client_model) in self.updates:
            total_weight_hha += client_samples
            for key, value in client_model.items():
                if "hha_backbone" in key:
                    if key in base_hha:
                        base_hha[key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        base_hha[key] = client_samples * value.type(torch.FloatTensor)
                elif "classifier" in key:
                    if key in base_decoder:
                        base_decoder[key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        base_decoder[key] = client_samples * value.type(torch.FloatTensor)

        for (client_samples, client_model) in self.updates_rgb:
            total_weight_rgb += client_samples
            for key, value in client_model.items():
                if "rgb_backbone" in key:
                    if key in base_rgb:
                        base_rgb[key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        base_rgb[key] = client_samples * value.type(torch.FloatTensor)
                elif "classifier" in key:
                    if key in base_decoder:
                        base_decoder[key] += client_samples * value.type(torch.FloatTensor)
                    else:
                        base_decoder[key] = client_samples * value.type(torch.FloatTensor)

        averaged_sol_n_hha = copy.deepcopy(self.hha_backbone_params_dict)
        averaged_sol_n_rgb = copy.deepcopy(self.rgb_backbone_params_dict)
        averaged_sol_n_decoder = copy.deepcopy(self.classifier_params_dict)

        for key, value in base_hha.items():
            if total_weight_hha != 0:
                new_key = key.replace("module.hha_backbone.", "")

                averaged_sol_n_hha[new_key] = value.to(self.local_rank) / total_weight_hha

        for key, value in base_rgb.items():
            if total_weight_rgb != 0:
                new_key = key.replace("module.rgb_backbone.", "")
                averaged_sol_n_rgb[new_key] = value.to(self.local_rank) / total_weight_rgb

        for key, value in base_decoder.items():
            if total_weight_hha != 0 or total_weight_rgb != 0:
                new_key = key.replace("module.classifier.", "")
                averaged_sol_n_decoder[new_key] = value.to(self.local_rank) / (total_weight_hha + total_weight_rgb)

        return averaged_sol_n_hha, averaged_sol_n_rgb, averaged_sol_n_decoder

    def _server_opt(self, pseudo_gradient):
        for n, p in self.model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]
        self.optimizer.step()
        bn_layers = \
            OrderedDict({k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.model.load_state_dict(bn_layers, strict=False)

    def _server_opt_rgb(self, pseudo_gradient):
        for n, p in self.model_rgb.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]
        self.optimizer_rgb.step()
        bn_layers = \
            OrderedDict({k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.model_rgb.load_state_dict(bn_layers, strict=False)

    def _get_model_total_grad(self):
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad = total_norm ** 0.5
        self.writer.write(f"total grad norm HHA: {round(total_grad, 2)}")
        return total_grad

    def _get_model_rgb_total_grad(self):
        total_norm = 0
        for name, p in self.model_rgb.named_parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad_rgb = total_norm ** 0.5
        self.writer.write(f"total grad norm RGB: {round(total_grad_rgb, 2)}")
        return total_grad_rgb

    def update_model(self):

        print("AGGREGATION: END OF THE ROUND")
        if self.args.mm_setting=="first":
            averaged_sol_n_rgb = self._aggregation_rgb()
            averaged_sol_n = self._aggregation()

            # self.optimizer_rgb is NONE
            if self.optimizer_rgb is not None:
                self._server_opt_rgb(averaged_sol_n_rgb)
                self.total_grad_rgb = self._get_model_rgb_total_grad()
            else:
                #entra qui
                self.model_rgb.load_state_dict(averaged_sol_n_rgb)

            if self.optimizer is not None:
                self._server_opt(averaged_sol_n)
                self.total_grad = self._get_model_total_grad()
            else:
                #entra qui
                self.model.load_state_dict(averaged_sol_n)

            self.model_rgb_params_dict = copy.deepcopy(self.model_rgb.state_dict())
            self.model_params_dict = copy.deepcopy(self.model.state_dict())

            self.updates_rgb = []
            self.updates = []

        elif self.args.mm_setting == "second":
            averaged_sol_n_hha, averaged_sol_n_rgb, averaged_sol_n_decoder = self._aggregation_second_exp()

            self.model.module.rgb_backbone.load_state_dict(averaged_sol_n_rgb)
            self.model.module.hha_backbone.load_state_dict(averaged_sol_n_hha)
            self.model.module.classifier.load_state_dict(averaged_sol_n_decoder)

            if self.optimizer is not None:
                print("da fare")
            self.rgb_backbone_params_dict = copy.deepcopy(self.model.module.rgb_backbone.state_dict())
            self.hha_backbone_params_dict = copy.deepcopy(self.model.module.hha_backbone.state_dict())
            self.classifier_params_dict = copy.deepcopy(self.model.module.classifier.state_dict())

            self.updates_rgb = []
            self.updates = []
        else:
            averaged_sol_n = self._aggregation()

            if self.optimizer is not None:
                self._server_opt(averaged_sol_n)
                self.total_grad = self._get_model_total_grad()
            else:
                self.model.load_state_dict(averaged_sol_n)
            self.model_params_dict = copy.deepcopy(self.model.state_dict())

            self.updates = []