from metrics import StreamSegMetrics
from federated.trainers.trainer import Trainer


class OracleTrainer(Trainer):

    def __init__(self, args, writer, device, rank, world_size):
        super().__init__(args, writer, device, rank, world_size)

    @staticmethod
    def set_metrics(writer, num_classes):
        writer.write("Setting up metrics...")
        metrics = {
            'test': StreamSegMetrics(num_classes, 'test'),
            'partial_train': StreamSegMetrics(num_classes, 'partial_train'),
            'eval_train': StreamSegMetrics(num_classes, 'eval_train')
        }
        writer.write("Done.")
        return metrics

    def handle_ckpt_step(self):
        return None, None, self.checkpoint_step, None

    def perform_fed_oracle_training(self, partial_train_metric, eval_train_metric, test_metric, partial_train_metric_2,
                                    eval_train_metric_2, test_metric_2, max_scores=None, max_scores_2=None):
        if max_scores is None:
            if self.args.mm_setting=="first":
                max_scores = [0]*len(self.target_test_clients)
                max_scores_2 = [0]*len(self.target_test_clients_2)
            else:
                max_scores = [0]*len(self.target_test_clients)


        for r in range(self.ckpt_round, self.args.num_rounds):
            self.writer.write(f'ROUND {r + 1}/{self.args.num_rounds}: '
                              f'Training {self.args.clients_per_round} Clients...')

            self.server.select_clients(r, self.target_train_clients, num_clients=self.args.clients_per_round)

            # primo esperimento
            if self.args.mm_setting=="first":
                losses, losses_2 = self.server.train_clients(partial_metric=partial_train_metric,  partial_metric_2=partial_train_metric_2)

                if len(losses) != 0:
                    self.plot_train_metric(r, partial_train_metric, losses)
                partial_train_metric.reset()

                if len(losses_2) != 0:
                    self.plot_train_metric(r, partial_train_metric_2, losses_2)
                partial_train_metric_2.reset()

                print("QUI ARRIVA ALL'UPDATE")
                self.server.update_model()
                self.model_rgb.load_state_dict(self.server.model_rgb_params_dict)
                self.save_model_rgb(r + 1, optimizer=self.server.optimizer_rgb)
                self.model.load_state_dict(self.server.model_params_dict)
                self.save_model(r + 1, optimizer=self.server.optimizer)

                if (r + 1) % self.args.eval_interval == 0 and \
                        self.all_target_client.loader.dataset.ds_type not in ('unsupervised',):
                    self.test([self.all_target_client], eval_train_metric, r, 'ROUND', self.get_fake_max_scores(False, 1),
                                  cl_type='target')
                    self.test([self.all_target_client_2], eval_train_metric_2, r, 'ROUND', self.get_fake_max_scores(False, 1),
                                  cl_type='target')

                if (r + 1) % self.args.test_interval == 0 or (r + 1) == self.args.num_rounds:
                    # sembra entrare solo qui, self.test si riferisce a general_trainer
                    print("Case:  ", self.target_test_clients[0].format_client)
                    max_scores, _ = self.test(self.target_test_clients, test_metric, r, 'ROUND', max_scores,
                                                  cl_type='target')
                    print("Case:  ", self.target_test_clients_2[0].format_client)
                    max_scores_2, _ = self.test(self.target_test_clients_2, test_metric_2, r, 'ROUND', max_scores_2,
                                                    cl_type='target')
            # secondo esperimento
            elif self.args.mm_setting == "second":
                losses = self.server.train_clients(partial_metric=partial_train_metric)
                self.plot_train_metric(r, partial_train_metric, losses)
                partial_train_metric.reset()

                print("QUI ARRIVA ALL'UPDATE")
                self.server.update_model()
                self.model.module.rgb_backbone.load_state_dict(self.server.rgb_backbone_params_dict)
                self.model.module.hha_backbone.load_state_dict(self.server.hha_backbone_params_dict)
                self.model.module.classifier.load_state_dict(self.server.classifier_params_dict)
                #self.save_model(r + 1, optimizer=self.server.optimizer)

                if (r + 1) % self.args.eval_interval == 0 and \
                        self.all_target_client.loader.dataset.ds_type not in ('unsupervised',):
                    self.test([self.all_target_client], eval_train_metric, r, 'ROUND', self.get_fake_max_scores(False, 1),
                                  cl_type='target')

                if (r + 1) % self.args.test_interval == 0 or (r + 1) == self.args.num_rounds:
                    #self.test si riferisce a general_trainer
                    max_scores, _ = self.test(self.target_test_clients, test_metric, r, 'ROUND', max_scores,
                                                  cl_type='target')

            #caso base + terzo
            else:
                losses = self.server.train_clients(partial_metric=partial_train_metric)
                self.plot_train_metric(r, partial_train_metric, losses)
                partial_train_metric.reset()

                print("QUI ARRIVA ALL'UPDATE")
                self.server.update_model()
                self.model.load_state_dict(self.server.model_params_dict)
                self.save_model(r + 1, optimizer=self.server.optimizer)

                if (r + 1) % self.args.eval_interval == 0 and \
                        self.all_target_client.loader.dataset.ds_type not in ('unsupervised',):
                    self.test([self.all_target_client], eval_train_metric, r, 'ROUND', self.get_fake_max_scores(False, 1),
                                  cl_type='target')

                if (r + 1) % self.args.test_interval == 0 or (r + 1) == self.args.num_rounds:
                    # sembra entrare solo qui, self.test si riferisce a general_trainer
                    max_scores, _ = self.test(self.target_test_clients, test_metric, r, 'ROUND', max_scores,
                                                  cl_type='target')


        if self.args.mm_setting == "first":
            return max_scores, max_scores_2
        else:
            return max_scores


    def train(self):
        if self.args.mm_setting == "first":
            return self.perform_fed_oracle_training(
                    partial_train_metric=self.metrics['partial_train'],
                    eval_train_metric=self.metrics['eval_train'],
                    test_metric=self.metrics['test'],
                    partial_train_metric_2=self.metrics_2['partial_train'],
                    eval_train_metric_2=self.metrics_2['eval_train'],
                    test_metric_2=self.metrics_2['test'])

        else:
            return self.perform_fed_oracle_training(
                    partial_train_metric=self.metrics['partial_train'],
                    eval_train_metric=self.metrics['eval_train'],
                    test_metric=self.metrics['test'],
                    partial_train_metric_2=None,
                    eval_train_metric_2=None,
                    test_metric_2=None)


