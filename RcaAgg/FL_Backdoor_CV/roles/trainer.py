import os
os.environ["OMP_NUM_THREADS"] = "1"
import random
import time
import sys
import numpy as np
import torch.backends.cudnn
import yaml

from FL_Backdoor_CV.image_helper import ImageHelper
from configs import args

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from FL_Backdoor_CV.roles.client import Client
from FL_Backdoor_CV.roles.server import Server

from torchvision.utils import save_image
import matplotlib.pyplot as plt

torch.manual_seed(1)
torch.cuda.manual_seed(1)

random.seed(0)
np.random.seed(0)
class Logger(object):
    def __init__(self, filename="output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
def check_params(params):
    assert params['participant_sample_size'] <= params['participant_population']
    assert params['number_of_adversaries'] <= params['participant_sample_size']

class Trainer:
    def __init__(self):
        print(f"We are conduct federated training using {args.aggregation_rule} on {args.dataset}... \n")

        # === Load yaml file ===
        with open(f'../{args.params}', 'r', encoding='utf-8') as f:
            params_loaded = yaml.load(f, Loader=Loader)

        # === Add additional fields to the loaded params based on args ===
        params_loaded.update(vars(args))
        if params_loaded['dataset'] == 'cifar10' or \
                params_loaded['dataset'] == 'emnist' or \
                params_loaded['dataset'] == 'fmnist':
            dataset_name = params_loaded['dataset']
            if os.path.isdir(f'./data/{dataset_name}/'):
                params_loaded['data_folder'] = f'./data/{dataset_name}'
            params_loaded['participant_clean_data'] = random.sample(
                range(params_loaded['participant_population'])[args.number_of_adversaries:], 150)
        else:
            raise ValueError('Unrecognized dataset')
        self.params_loaded = params_loaded

        # === Check parameters ===
        check_params(self.params_loaded)

        if args.attack_mode.upper() == 'COMBINE2':
            args.multi_objective_num = 3
            print(f"COMBINE2 mode: set multi_objective_num to {args.multi_objective_num}")
        # === load data ===
        start_time = time.time()
        print(f"--------------------- L O A D I N G - D A T A ---------------------")
        helper = ImageHelper(params=self.params_loaded)
        helper.load_data()
        helper.load_distributed_data()
        helper.load_benign_data()
        helper.load_poison_data()
        plot = False
        if plot:
            for batch in helper.poisoned_train_data:
                X, y = helper.get_poison_batch(batch)
                for i in range(1):
                    print(y[i])
                    if args.dataset.lower() in ['emnist', 'fmnist']:
                        plt.imshow(X[i].reshape(28, 28))
                        if args.dataset.lower() == 'emnist':
                            plt.savefig('./emnist.pdf', bbox_inches='tight')
                        if args.dataset.lower() == 'fmnist':
                            plt.savefig('./fmnist.pdf', bbox_inches='tight')
                        plt.show()
                    elif args.dataset.lower() == 'cifar10':
                        save_image(X[i], './cifar10.pdf')
                break
            exit(0)
        print(f"Time spent: {time.time() - start_time: .4f} seconds!")
        print(f"--------------------- L O A D E D ! ---------------------")
        # === create clients ===
        clients = [Client(_id, local_data, local_data_size)
                   for _id, (local_data, local_data_size) in enumerate(zip(helper.train_data, helper.local_data_sizes))]
        # === mal indices ===
        adversary_list = list()
        if args.is_poison:
            normal_adversaries = list(range(args.number_of_adversaries))
            adaptive_start = args.number_of_adversaries
            adaptive_end = args.number_of_adversaries + args.adaptive_adversaries
            adaptive_adversaries = list(range(adaptive_start, adaptive_end))
            adversary_list = normal_adversaries + adaptive_adversaries
            if args.attack_mode.lower() == 'dba':
                selection = list(range(args.dba_trigger_num))
                adversary_index = selection * int(args.number_of_adversaries / args.dba_trigger_num)
                for i in range(args.number_of_adversaries % args.dba_trigger_num):
                    adversary_index.append([i])
                assert len(adversary_list) == len(adversary_index)
                for idx, ind in zip(adversary_list, adversary_index):
                    clients[idx].malicious = True
                    clients[idx].adversarial_index = ind
                    if idx in adaptive_adversaries:
                        clients[idx].is_adaptive_adversary = True
            elif args.attack_mode.lower() == 'combine':
                selection = list(range(args.multi_objective_num))
                adversary_index = selection * int(args.number_of_adversaries / args.multi_objective_num)
                for i in range(args.number_of_adversaries % args.multi_objective_num):
                    adversary_index.append(selection[i])
                assert len(adversary_list) == len(adversary_index)
                for idx, ind in zip(adversary_list, adversary_index):
                    clients[idx].malicious = True
                    clients[idx].adversarial_index = ind
                    if idx in adaptive_adversaries:
                        clients[idx].is_adaptive_adversary = True
            elif args.attack_mode.lower() == 'combine2':
                selection = list(range(args.multi_objective_num))
                adversary_index = selection * int(args.number_of_adversaries / args.multi_objective_num)
                for i in range(args.number_of_adversaries % args.multi_objective_num):
                    adversary_index.append(selection[i])
                assert len(adversary_list) == len(adversary_index)
                for idx, ind in zip(adversary_list, adversary_index):
                    clients[idx].malicious = True
                    clients[idx].adversarial_index = ind
                    if idx in adaptive_adversaries:
                        clients[idx].is_adaptive_adversary = True
            else:
                for idx in adversary_list:
                    clients[idx].malicious = True
                    if idx in adaptive_adversaries:
                        clients[idx].is_adaptive_adversary = True

        # === create server ===
        self.server = Server(helper, clients, adversary_list)
        for client in clients:
            client.server = self.server
        if args.is_poison:
            print(f"Malicious clients configuration:")
            print(
                f"  Normal adversaries: {len(normal_adversaries)} (ID: {min(normal_adversaries)}-{max(normal_adversaries)})")
            if adaptive_adversaries:
                print(
                    f"  Adaptive adversaries: {len(adaptive_adversaries)} (ID: {min(adaptive_adversaries)}-{max(adaptive_adversaries)})")
        # === result ===
        self.results = dict()
        if args.attack_mode.lower() in ['combine', 'combine2']:
            self.results = {'loss': [], 'accuracy': []}
            for i in range(args.multi_objective_num):
                self.results[f'poison_loss_{i}'] = list()
                self.results[f'poison_accuracy_{i}'] = list()
        else:
            print("=== Entering single-backdoor branch1 ===")
            self.results = {'loss': [], 'accuracy': [], 'poison_loss': [], 'poison_accuracy': []}

    def train(self):
        assert self.server is not None
        # === save path ===
        localtime = time.localtime(time.time())
        path = f"{args.aggregation_rule}_{args.attack_mode}_" \
               f"{localtime[1]:02}{localtime[2]:02}{localtime[3]:02}{localtime[4]:02}"
        saved_model_path = os.path.join('../saved_models/Revision_1', path)
        saved_results_path = os.path.join('../results/Revision_1', args.aggregation_rule)
        res_path = os.path.join(saved_results_path, f"{args.dataset}_{path}.pt")
        arg_path = os.path.join(saved_results_path, f"{args.dataset}_{path}.txt")
        if not os.path.exists(saved_results_path):
            os.makedirs(saved_results_path)
        with open(arg_path, 'w') as f:
            for eachArg, value in args.__dict__.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')

        # === federated training process ===
        start_round = self.server.current_round
        end_round = args.rounds
        if args.is_poison:
            end_round = start_round + args.retrain_rounds + args.windows
        print(f"Total rounds: {end_round - start_round}")
        for _ in range(start_round, end_round):
            start_time = time.time()
            # === participants selection phase ===
            self.server.select_participants()

            # === training and aggregating phase ===
            if args.aggregation_rule == 'fedcie':
                if self.server.current_round > 500:
                    if args.dataset.lower() == 'cifar10':
                        global_lr = args.global_lr * args.global_lr_decay ** (self.server.current_round - 500)
                        self.server.train_and_aggregate(global_lr=global_lr)
                else:
                    self.server.train_and_aggregate(global_lr=args.global_lr)
            else:
                self.server.train_and_aggregate(global_lr=args.global_lr)

            if args.aggregation_rule.lower() == 'rcaagg' and self.server.current_round % 1 == 0:
                self.server.log_reputation_stats()

            # === evaluation phase ===
            test_loss, test_acc = self.server.validate()
            self.results['accuracy'].append(test_acc)
            self.results['loss'].append(test_loss)

            if args.attack_mode.lower() in ['combine', 'combine2']:
                test_l_acc = self.server.validate_poison()
                for i, l_acc in enumerate(test_l_acc):
                    poison_loss, poison_accuracy = l_acc
                    self.results[f'poison_accuracy_{i}'].append(poison_accuracy)
                    self.results[f'poison_loss_{i}'].append(poison_loss)
                print(f"[Round: {self.server.current_round: 04}], "
                      f"Accuracy on testset: {test_acc: .4f}, "
                      f"Loss on testset: {test_loss: .4f}.")
                for i in range(args.multi_objective_num):
                    if i == 0:
                        print(f"                 Poison accuracy (o1): {test_l_acc[0][1]: .4f}.", end='   =========   ')
                    elif i == 1:
                        print(f"Poison accuracy (o2): {test_l_acc[1][1]: .4f}.")
                    elif i == 2:
                        print(f"                 Poison accuracy (o3): {test_l_acc[2][1]: .4f}.", end='   =========   ')
                    elif i == 3:
                        print(f"Poison accuracy (o4): {test_l_acc[3][1]: .4f}.")
            else:
                test_poison_loss, test_poison_acc = self.server.validate_poison()
                self.results['poison_accuracy'].append(test_poison_acc)
                self.results['poison_loss'].append(test_poison_loss)
                print(f"[Round: {self.server.current_round: 04}], "
                      f"Accuracy on testset: {test_acc: .4f}, "
                      f"Loss on testset: {test_loss: .4f}. <---> "
                      f"Poison accuracy: {test_poison_acc: .4f}, "
                      f"Poison loss: {test_poison_loss: .4f}."
                      )

            print(f"Time spent: {time.time() - start_time: .4f} seconds, "
                  f"Estimated time required to complete the training: "
                  f"{(time.time() - start_time) * (end_round - self.server.current_round) / 3600: .4f}"
                  f" hours.\n")

            # === save the model every args.record_step rounds ===
            if self.server.current_round % args.record_step == 0:
                if not os.path.exists(saved_model_path):
                    os.makedirs(saved_model_path)
                model_file = os.path.join(saved_model_path,
                                          f"{args.aggregation_rule}_{self.server.current_round}.pth")

                self.server.save_model_with_reputation(model_file)

            if args.resume:
                model_path = os.path.join('../saved_models/Revision_1', args.resumed_name)
                self.server.load_model_with_reputation(model_path)

            # === save the results every args.record_res_step rounds ===
            if self.server.current_round % args.record_res_step == 0:
                torch.save(self.results, res_path)

if __name__ == '__main__':
    import datetime
    pid = os.getpid()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.aggregation_rule}_{args.attack_mode}_{args.dataset}_{timestamp}_{pid}"
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "output_log.txt")
    sys.stdout = Logger(log_file)
    trainer = Trainer()
    trainer.train()