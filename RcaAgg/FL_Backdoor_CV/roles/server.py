import copy
import os
import random
import re
from collections import defaultdict
import numpy as np
import sklearn.metrics.pairwise as smp
import torch
import torch.nn.functional as F
from scipy.linalg import eigh as largest_eigh
from scipy.stats import entropy
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from FL_Backdoor_CV.models.create_model import create_model
from FL_Backdoor_CV.roles.evaluation import test_cv, test_poison_cv
from configs import args
from FL_Backdoor_CV.roles.aggregation_rules import rcaagg
from FL_Backdoor_CV.roles.aggregation_rules import avg
from FL_Backdoor_CV.roles.aggregation_rules import foolsgold
from FL_Backdoor_CV.roles.aggregation_rules import flame
from FL_Backdoor_CV.roles.aggregation_rules import fltrust
from FL_Backdoor_CV.roles.aggregation_rules import robust_lr

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

class Server:
    def __init__(self, helper, clients, adversary_list):
        self.stop_reputation_update = False
        self.last_acc_mal_new = 0
        self.benign_mean_history = []
        self.benign_var_history = []
        self.malicious_mean_history = []
        self.malicious_var_history = []
        self.reputation_history = {}
        self.acc_mal_new_history = []
        self.lowest_new_reputation_clients_history = []
        self.conspiracy_graph = self.ConspiracyGraph()
        # === model ===
        print(os.getcwd())
        if args.resume:
            model_path = os.path.join('../saved_models/Revision_1', args.resumed_name)
            loaded_data = torch.load(model_path, map_location=args.device, weights_only=False)
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:

                self.model = create_model()
                self.model.load_state_dict(loaded_data['model_state_dict'])
                if 'reputations' in loaded_data:
                    reputations = loaded_data['reputations']
                    for client_id, reputation in reputations.items():
                        if client_id < len(self.clients):
                            self.clients[client_id].reputation = reputation
                    print(f"Loaded reputations for {len(reputations)} clients")

                if 'current_round' in loaded_data:
                    self.current_round = loaded_data['current_round']
                if 'poison_rounds' in loaded_data:
                    self.poison_rounds = loaded_data['poison_rounds']
            else:
                self.model = loaded_data
        else:
            self.model = create_model()

        # === gradient correction ===
        self.previous_models = []
        if args.gradient_correction:
            previous_model = copy.deepcopy(self.model)
            previous_model.load_state_dict(self.model.state_dict())
            self.previous_models.append(previous_model)

        self.clients = clients
        self.participants = None
        self.adversary_list = adversary_list
        self.benign_indices = list(set(list(range(args.participant_population))) - set(self.adversary_list))

        # === image helper ===
        self.helper = helper
        # === whether resume ===
        self.current_round = 0
        # === Inherent recognition accuracy on poisoned data sets
        self.inheret_poison_acc = 0

        if args.resume:
            self.current_round = int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0])
            test_l, test_acc = self.validate()
            if args.attack_mode.lower() in ['combine', 'combine2']:
                test_l_acc = self.validate_poison()
                print(f"\n--------------------- T e s t - L o a d e d - M o d e l ---------------------")
                print(f"Accuracy on testset: {test_acc: .4f}, Loss on testset: {test_l: .4f}.")
                for i in range(args.multi_objective_num):
                    if i == 0:
                        print(f"Poison accuracy (o1): {test_l_acc[0][1]: .4f}.", end='   =========   ')
                    elif i == 1:
                        print(f"Poison accuracy (o2): {test_l_acc[1][1]: .4f}.")
                    elif i == 2:
                        print(f"Poison accuracy (o3): {test_l_acc[2][1]: .4f}.", end='   =========   ')
                    elif i == 3:
                        print(f"Poison accuracy (o4): {test_l_acc[3][1]: .4f}.")
                    elif i == 4:
                        print(f"Poison accuracy (wall ---> bird): {test_l_acc[4][1]: .4f}.")
                    elif i == 5:
                        print(f"Poison accuracy (green car ---> bird): {test_l_acc[5][1]: .4f}.")
                    elif i == 6:
                        print(f"Poison accuracy (strip car ---> bird): {test_l_acc[6][1]: .4f}.")
                print(f"--------------------- C o m p l e t e ! ---------------------\n")
            else:
                test_poison_loss, test_poison_acc = self.validate_poison()
                self.inheret_poison_acc = test_poison_acc
                print(f"\n--------------------- T e s t - L o a d e d - M o d e l ---------------------\n"
                      f"Accuracy on testset: {test_acc: .4f}, "
                      f"Loss on testset: {test_l: .4f}. <---> "
                      f"Poison accuracy: {test_poison_acc: .4f}, "
                      f"Poison loss: {test_poison_loss: .4f}"
                      f"\n--------------------- C o m p l e t e ! ---------------------\n")
        # === total data size ===
        self.total_size = 0

        # === whether poison ===
        self.poison_rounds = list()
        self.is_poison = args.is_poison
        if self.is_poison:
            # === give the poison rounds in the configuration ===
            if args.poison_rounds:
                assert isinstance(args.poison_rounds, str)
                self.poison_rounds = [int(i) for i in args.poison_rounds.split(',')]
            else:
                retrain_rounds = np.arange(self.current_round + 1 + args.windows,
                                           self.current_round + args.windows + args.retrain_rounds + 1)
                whether_poison = np.random.uniform(0, 1, args.retrain_rounds) >= (1 - args.poison_prob)
                self.poison_rounds = set((retrain_rounds * whether_poison).tolist())
                if 0 in self.poison_rounds:
                    self.poison_rounds.remove(0)
                self.poison_rounds = list(self.poison_rounds)
            args.poison_rounds = self.poison_rounds

            print(f"\n--------------------- P o i s o n - R o u n d s : {self.poison_rounds} ---------------------\n")
        else:
            print(f"\n--------------------- P o i s o n - R o u n d s : N o n e ! ---------------------\n")

        # === root dataset ===
        self.root_dataset = None
        if args.aggregation_rule.lower() == 'fltrust':
            self.root_dataset = self.helper.load_root_dataset()

    class ConspiracyGraph:
        def __init__(self):
            self.conspiracy_edges = {}
            self.client_participation_rounds = {}
            self.client_malicious_rounds = {}

    def compute_all_client_suspect_scores(self):
        if self.conspiracy_graph is None:
            return {client_id: 0.0 for client_id in range(args.participant_population)}

        suspect_scores = {}
        for client_id in range(args.participant_population):

            edge_weight_sum = 0
            for edge_key, weight in self.conspiracy_graph.conspiracy_edges.items():
                u, v = map(int, edge_key.split('-'))
                if client_id == u or client_id == v:
                    edge_weight_sum += weight

            participation_rounds = self.conspiracy_graph.client_participation_rounds.get(client_id, 1)

            suspect_score = edge_weight_sum / participation_rounds if participation_rounds > 0 else 0.0
            suspect_scores[client_id] = suspect_score
        return suspect_scores

    def select_participants(self):
        self.current_round += 1
        self.total_size = 0

        if self.stop_reputation_update:

            print(f"Select using weighted by reputation points (T={args.softmax_temperature})")
            self._select_participants_by_reputation()
        else:

            print(f"Use the original random selection")
            self._select_participants_original()

        self._log_participant_selection()

    def _select_participants_original(self):
        if args.random_compromise:
            if self.current_round in self.poison_rounds:
                self.participants = random.sample(range(args.participant_population), args.participant_sample_size)
            else:
                self.participants = random.sample(range(args.number_of_adversaries, args.participant_population),
                                                  args.participant_sample_size)
        else:
            if self.current_round in self.poison_rounds:
                if args.attack_mode.lower() == 'dba':
                    candidates = list()
                    adversarial_index = self.poison_rounds.index(self.current_round) % args.dba_trigger_num
                    for client_id in self.adversary_list:
                        if self.clients[client_id].adversarial_index == adversarial_index:
                            candidates.append(client_id)
                    self.participants = candidates + random.sample(
                        self.benign_indices, args.participant_sample_size - len(candidates))
                else:
                    self.participants = self.adversary_list + random.sample(
                        self.benign_indices, args.participant_sample_size - len(self.adversary_list))
            else:
                self.participants = random.sample(self.benign_indices, args.participant_sample_size)

    def _log_probability_distribution(self, client_ids, reputations, probabilities):

        client_reputation_pairs = list(zip(client_ids, reputations, probabilities))
        sorted_by_reputation = sorted(client_reputation_pairs, key=lambda x: x[1], reverse=True)

        n_clients = len(client_ids)

        percentiles = [10, 25, 50, 75, 90]
        percentile_probs = {}

        for p in percentiles:
            count = int(n_clients * p / 100)
            percentile_probs[p] = sum(prob for _, _, prob in sorted_by_reputation[:count])

        print("The client selection probability grouped by the new reputation value:")
        for p in percentiles:
            print(f"  {p}%: {percentile_probs[p]:.3f}")

    def _log_participant_selection(self):

        for client_id in self.participants:
            self.total_size += self.clients[client_id].local_data_size

        malicious_participants = [client_id for client_id in self.participants if
                                  client_id in self.adversary_list]
        benign_participants = [client_id for client_id in self.participants if
                               client_id not in self.adversary_list]

        malicious_count = len(malicious_participants)
        benign_count = len(benign_participants)

        sorted_participants = sorted(self.participants)
        sorted_malicious = sorted(malicious_participants)
        sorted_benign = sorted(benign_participants)

        print(f"=== Round {self.current_round} Participant Selection ===")
        print(f"Selection mode: {'Reputation-weighted' if self.stop_reputation_update else 'Original random'}")
        print(f"Reputation update: {'Stopped' if self.stop_reputation_update else 'Active'}")
        print(f"Poison round: {'Yes' if self.current_round in self.poison_rounds else 'No'}")
        print(f"Total participants: {len(sorted_participants)}")
        print(f"All participants (sorted): {sorted_participants}")
        print(f"Benign participants ({benign_count}): {sorted_benign}")
        if malicious_count > 0:
            print(f"Malicious participants ({malicious_count}): {sorted_malicious}")
        else:
            print(f"Malicious participants: None")

        participant_new_reps = [self.clients[client_id].new_reputation for client_id in self.participants]
        if participant_new_reps:
            avg_new_rep = sum(participant_new_reps) / len(participant_new_reps)
            min_new_rep = min(participant_new_reps)
            max_new_rep = max(participant_new_reps)
            print(f"Participants new reputation - Avg: {avg_new_rep:.1f}, Min: {min_new_rep:.1f}, Max: {max_new_rep:.1f}")
        print(f"Total data size: {self.total_size}")
        print("=" * 50)

    def update_reputation_by_update_quality(self, credit_scores, participant_indices):
        if credit_scores is None or len(credit_scores) < 3:
            return
        client_score_pairs = list(zip(participant_indices, credit_scores))
        sorted_clients = sorted(client_score_pairs, key=lambda x: x[1])
        bottom_3_clients = [client_id for client_id, _ in sorted_clients[:3]]
        top_3_clients = [client_id for client_id, _ in sorted_clients[-3:]]
        for client_id in bottom_3_clients:
            old_reputation = self.clients[client_id].new_reputation
            self.clients[client_id].new_reputation -= 0.1
            self.clients[client_id].new_reputation = max(-500.0, min(500.0, self.clients[client_id].new_reputation))
            print(
                f"Client {client_id}: low3 -> -0.1 ({old_reputation:.3f} -> {self.clients[client_id].new_reputation:.3f})")
        for client_id in top_3_clients:
            old_reputation = self.clients[client_id].new_reputation
            self.clients[client_id].new_reputation += 0.1
            self.clients[client_id].new_reputation = max(-500.0, min(500.0, self.clients[client_id].new_reputation))
            print(
                f"Client {client_id}: high3 -> +0.1 ({old_reputation:.3f} -> {self.clients[client_id].new_reputation:.3f})")

    def train_and_aggregate(self, global_lr):
        is_attack_round = self.current_round in self.poison_rounds
        trained_models = dict()

        participant_clients = [self.clients[client_id] for client_id in self.participants]
        client_reputations = [client.reputation for client in participant_clients]

        # === trained local models ===
        trained_models = dict()
        param_updates = list()
        trained_params = list()
        for client_id in self.participants:
            local_model = copy.deepcopy(self.model)
            local_model.load_state_dict(self.model.state_dict())
            trained_local_model = self.clients[client_id].local_train(local_model, self.helper, self.current_round)
            if args.aggregation_rule.lower() == 'fltrust':
                param_updates.append(parameters_to_vector(trained_local_model.parameters()) - parameters_to_vector(
                    self.model.parameters()))
            elif args.aggregation_rule.lower() == 'flame':
                trained_param = parameters_to_vector(trained_local_model.parameters()).detach().cpu()
                trained_params.append(trained_param)

            for name, param in trained_local_model.state_dict().items():
                if name not in trained_models:
                    trained_models[name] = param.data.view(1, -1)
                else:
                    trained_models[name] = torch.cat((trained_models[name], param.data.view(1, -1)),
                                                     dim=0)

        model_updates = dict()
        for (name, param), local_param in zip(self.model.state_dict().items(), trained_models.values()):
            model_updates[name] = local_param.data - param.data.view(1, -1)
            if args.attack_mode.lower() in ['mr', 'dba', 'flip', 'edge_case', 'neurotoxin', 'combine']:
                if 'num_batches_tracked' not in name:
                    for i in range(args.participant_sample_size):
                        if self.clients[self.participants[i]].malicious:
                            mal_boost = 1
                            if args.is_poison:
                                if args.mal_boost:
                                    if args.attack_mode.lower() in ['mr', 'flip', 'edge_case', 'neurotoxin', 'combine', 'combine2']:
                                        mal_boost = args.mal_boost / args.number_of_adversaries
                                    elif args.attack_mode.lower() == 'dba':
                                        mal_boost = args.mal_boost / (args.number_of_adversaries / args.dba_trigger_num)
                                else:
                                    if args.attack_mode.lower() in ['mr', 'flip', 'edge_case', 'neurotoxin', 'combine', 'combine2']:
                                        mal_boost = args.participant_sample_size / args.number_of_adversaries
                                    elif args.attack_mode.lower() == 'dba':
                                        mal_boost = args.participant_sample_size / \
                                                    (args.number_of_adversaries / args.dba_trigger_num)
                            model_updates[name][i] *= (mal_boost / args.global_lr)

        # === aggregate ===
        global_update = None
        if args.aggregation_rule.lower() == 'avg':
            global_update = avg(model_updates)
        elif args.aggregation_rule.lower() == 'rcaagg':
            client_new_reputations = {client.client_id: client.new_reputation for client in self.clients}
            reputation_dict = {client.client_id: client.reputation for client in self.clients}
            all_suspect_scores = self.compute_all_client_suspect_scores()
            global_update, credit_scores, client_cluster_sizes, weighted_variance_rank_head, weighted_variance_rank_tail, global_sim_idxs, avg_reputation_rank_head, avg_reputation_rank_tail,self.conspiracy_graph = rcaagg(
                model_updates,
                current_round=self.current_round,
                client_reputations=client_reputations,
                participant_indices=self.participants,
                reputation_dict=reputation_dict,
                client_new_reputations=client_new_reputations,
                stop_reputation_update=self.stop_reputation_update,
                conspiracy_graph=self.conspiracy_graph,
                all_suspect_scores=all_suspect_scores,
            )

            bottom_10_percent_clients = []
            bottom_10_30_percent_clients = []
            cluster_adjustment_info_sorted = []

            if credit_scores is not None and global_sim_idxs is not None:

                mean_credit_score = np.mean(credit_scores)
                cluster_adjustment_info = []
                for cluster in global_sim_idxs:
                    cluster_adjustments = []
                    for global_client_id in cluster:

                        if global_client_id in self.participants:
                            index_in_participants = self.participants.index(global_client_id)
                            client_credit = credit_scores[index_in_participants]
                            adjustment = client_credit - mean_credit_score
                            cluster_adjustments.append(adjustment)

                    cluster_adjustment_mean = np.mean(cluster_adjustments) if cluster_adjustments else 0.0

                    cluster_adjustment_info.append({
                        'cluster': cluster,
                        'adjustment_mean': cluster_adjustment_mean,
                        'client_ids': cluster
                    })

                cluster_adjustment_info_sorted = sorted(
                    cluster_adjustment_info,
                    key=lambda x: x['adjustment_mean']
                )

            if args.penalty:

                total_clusters = len(cluster_adjustment_info_sorted)

                bottom_10_percent_count = max(1, round(total_clusters * 0.1))
                bottom_10_30_percent_count = max(1, round(total_clusters * 0.2))

                bottom_10_percent_clusters = cluster_adjustment_info_sorted[:bottom_10_percent_count]
                bottom_10_30_percent_clusters = cluster_adjustment_info_sorted[
                                                bottom_10_percent_count:bottom_10_percent_count + bottom_10_30_percent_count]

                bottom_10_percent_negative_clusters = [cluster for cluster in bottom_10_percent_clusters if
                                                       cluster['adjustment_mean'] < 0]
                bottom_10_30_percent_negative_clusters = [cluster for cluster in bottom_10_30_percent_clusters if
                                                          cluster['adjustment_mean'] < 0]

                bottom_10_percent_clients = []
                for cluster_info in bottom_10_percent_negative_clusters:
                    bottom_10_percent_clients.extend(cluster_info['client_ids'])
                bottom_10_30_percent_clients = []
                for cluster_info in bottom_10_30_percent_negative_clusters:
                    bottom_10_30_percent_clients.extend(cluster_info['client_ids'])

            if self.current_round >= args.switch_rounds and not self.stop_reputation_update:
                self.stop_reputation_update = True

            if not self.stop_reputation_update:
                for client_id in self.participants:
                    self.clients[client_id].update_nnew_reputation(weighted_variance_rank_head,
                                                                   weighted_variance_rank_tail,
                                                                   avg_reputation_rank_head,
                                                                   avg_reputation_rank_tail)

            if self.stop_reputation_update and credit_scores is not None and args.reputation_by_quality:
                self.update_reputation_by_update_quality(credit_scores, self.participants)

            if credit_scores is not None:

                normalized_scores = [max(0.0, min(1.0, score)) for score in credit_scores]
                mean_normalized_score = np.mean(normalized_scores) if normalized_scores else 0.5

                all_adjustments = [score - mean_normalized_score for score in normalized_scores]

                for i, (client_id, credit_score) in enumerate(zip(self.participants, credit_scores)):

                    cluster_size = client_cluster_sizes.get(client_id, 1)
                    self.clients[client_id].update_reputation(
                        credit_score,
                        mean_normalized_score,
                        all_adjustments,
                        cluster_size,
                        bottom_10_percent_clients,
                        bottom_10_30_percent_clients
                    )

        elif args.aggregation_rule.lower() == 'foolsgold':
            global_update = foolsgold(model_updates)
        elif args.aggregation_rule.lower() == 'flame':
            current_model_param = parameters_to_vector(self.model.parameters()).detach().cpu()
            global_param, global_update = flame(trained_params, current_model_param, model_updates)
            vector_to_parameters(global_param, self.model.parameters())
            model_param = self.model.state_dict()
            for name, param in model_param.items():
                if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
                    model_param[name] = param.data + global_update[name].view(param.size())
            self.model.load_state_dict(model_param)
            return
        elif args.aggregation_rule.lower() == 'fltrust':
            if self.current_round > 500:
                lr = args.local_lr * args.local_lr_decay ** ((self.current_round - 500) // args.decay_step)
            else:
                lr = args.local_lr
            model = copy.deepcopy(self.model)
            model.load_state_dict(self.model.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                        momentum=self.helper.params['momentum'],
                                        weight_decay=self.helper.params['decay'])
            epochs = self.helper.params['retrain_no_times']
            criterion = torch.nn.CrossEntropyLoss()
            for _ in range(epochs):
                for inputs, labels in self.root_dataset:
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()

            clean_param_update = parameters_to_vector(model.parameters()) - parameters_to_vector(
                self.model.parameters())

            global_update = fltrust(model_updates, param_updates, clean_param_update)
        elif args.aggregation_rule.lower() == 'rlr':
            global_update = robust_lr(model_updates)

        # === update the global model ===
        model_param = self.model.state_dict()
        for name, param in model_param.items():
            model_param[name] = param.data + global_lr * global_update[name].view(param.size())
        self.model.load_state_dict(model_param)

    def log_reputation_stats(self):
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        reputations = [client.reputation for client in self.clients]
        benign_reputations = [client.reputation for client in self.clients if not client.malicious]
        malicious_reputations = [client.reputation for client in self.clients if client.malicious]
        new_reputations = [client.new_reputation for client in self.clients]
        benign_new_reputations = [client.new_reputation for client in self.clients if not client.malicious]
        malicious_new_reputations = [client.new_reputation for client in self.clients if client.malicious]
        all_mean = np.mean(reputations)
        all_std = np.std(reputations)
        benign_mean = np.mean(benign_reputations) if benign_reputations else 0.0
        benign_std = np.std(benign_reputations) if benign_reputations else 0.0
        benign_var = benign_std ** 2
        malicious_mean = np.mean(malicious_reputations) if malicious_reputations else 0.0
        malicious_std = np.std(malicious_reputations) if malicious_reputations else 0.0
        malicious_var = malicious_std ** 2

        self.benign_mean_history.append(benign_mean)
        self.benign_var_history.append(benign_var)
        self.malicious_mean_history.append(malicious_mean)
        self.malicious_var_history.append(malicious_var)


        try:
            plot_dir = "./reputation_plots"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            if len(self.benign_mean_history) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Benign and Malicious Clients Statistics Across Rounds', fontsize=16)
                client_rounds = list(range(1, len(self.benign_mean_history) + 1))
                mean_diff_history = [b - m for b, m in zip(self.benign_mean_history, self.malicious_mean_history)]
                std_diff_history = [np.sqrt(b) - np.sqrt(m) for b, m in
                                    zip(self.benign_var_history, self.malicious_var_history)]
                axes[0, 0].plot(client_rounds, self.benign_mean_history, 'o-', color='blue', linewidth=2, markersize=4,
                                label='Benign')
                axes[0, 0].plot(client_rounds, self.malicious_mean_history, 'o-', color='red', linewidth=2,
                                markersize=4, label='Malicious')
                axes[0, 0].set_title('Mean Reputation Comparison')
                axes[0, 0].set_ylabel('Mean Reputation')
                axes[0, 0].legend()

                benign_std_history = [np.sqrt(var) for var in self.benign_var_history]
                malicious_std_history = [np.sqrt(var) for var in self.malicious_var_history]
                axes[0, 1].plot(client_rounds, benign_std_history, 'o-', color='green', linewidth=2, markersize=4,
                                label='Benign')
                axes[0, 1].plot(client_rounds, malicious_std_history, 'o-', color='orange', linewidth=2, markersize=4,
                                label='Malicious')
                axes[0, 1].set_title('Standard Deviation Comparison')
                axes[0, 1].set_ylabel('Standard Deviation')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                axes[1, 0].plot(client_rounds, mean_diff_history, 'o-', color='purple', linewidth=2, markersize=4)
                axes[1, 0].set_title('Mean Difference (Benign - Malicious)')
                axes[1, 0].set_ylabel('Mean Difference')
                axes[1, 0].set_xlabel('Round')
                axes[1, 0].grid(True, alpha=0.3)

                axes[1, 1].plot(client_rounds, std_diff_history, 'o-', color='brown', linewidth=2, markersize=4)
                axes[1, 1].set_title('Std Difference (Benign - Malicious)')
                axes[1, 1].set_ylabel('Std Difference')
                axes[1, 1].set_xlabel('Round')
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                client_stats_path = os.path.join(plot_dir, "client_stats_trend.png")
                plt.savefig(client_stats_path, dpi=300, bbox_inches='tight')
                plt.close()

            if len(self.acc_mal_new_history) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                acc_mal_new_rounds = list(range(1, len(self.acc_mal_new_history) + 1))

                ax.plot(acc_mal_new_rounds, self.acc_mal_new_history, 'o-', color='red', linewidth=2, markersize=6)
                ax.set_title(f'ACC_MAL Trend Based on NEW Reputation (Malicious Clients in Lowest {args.number_of_adversaries} NEW Reputation)')
                ax.set_xlabel('Round')
                ax.set_ylabel('ACC_MAL Count')
                if args.number_of_adversaries > 0:
                    ax.set_ylim(0, args.number_of_adversaries)
                else:
                    max_value = max(self.acc_mal_new_history) if self.acc_mal_new_history else 0
                    ax.set_ylim(0, max(1, max_value))
                ax.grid(True, alpha=0.3)

                if len(self.acc_mal_new_history) > 1:
                    z = np.polyfit(acc_mal_new_rounds, self.acc_mal_new_history, 1)
                    p = np.poly1d(z)
                    ax.plot(acc_mal_new_rounds, p(acc_mal_new_rounds), "r--", alpha=0.7, linewidth=1,
                            label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
                    ax.legend()

                plt.tight_layout()
                acc_mal_new_path = os.path.join(plot_dir, "acc_mal_new_trend.png")
                plt.savefig(acc_mal_new_path, dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            print(f"  Warning: Could not create reputation plots: {e}")

        if args.plot_reputation_distribution and self.current_round % args.plot_reputation_frequency == 0:
            try:
                plot_dir = "./reputation_plots"
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                fig, ax = plt.subplots(figsize=(12, 8))
                bins = 30
                if len(reputations) < 50:
                    bins = 15

                n, bins, patches = ax.hist(reputations, bins=bins, alpha=0.7, color='skyblue',
                                           edgecolor='black', linewidth=1.2, label='All Clients',
                                           density=True)

                if benign_reputations:
                    ax.hist(benign_reputations, bins=bins, alpha=0.5, color='green',
                            edgecolor='darkgreen', linewidth=1.0, label='Benign Clients',
                            density=True, histtype='step', linestyle='--')
                if malicious_reputations:
                    ax.hist(malicious_reputations, bins=bins, alpha=0.5, color='red',
                            edgecolor='darkred', linewidth=1.0, label='Malicious Clients',
                            density=True, histtype='step', linestyle='--')

                ax.axvline(x=all_mean, color='blue', linestyle='-', linewidth=2,
                           label=f'All Mean: {all_mean:.3f}')
                if benign_reputations:
                    ax.axvline(x=benign_mean, color='green', linestyle='--', linewidth=2,
                               label=f'Benign Mean: {benign_mean:.3f}')
                if malicious_reputations:
                    ax.axvline(x=malicious_mean, color='red', linestyle='--', linewidth=2,
                               label=f'Malicious Mean: {malicious_mean:.3f}')

                if all_std > 0:
                    ax.axvspan(all_mean - all_std, all_mean + all_std, alpha=0.1, color='blue',
                               label=f'±1 std: [{all_mean - all_std:.3f}, {all_mean + all_std:.3f}]')

                ax.set_title(f'Round {self.current_round} - Client Reputation Distribution\n'
                             f'Total Clients: {len(reputations)}, '
                             f'Benign: {len(benign_reputations)}, '
                             f'Malicious: {len(malicious_reputations)}',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel('Reputation Value', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                ax.grid(True, alpha=0.3)

                ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

                stats_text = (f'Statistics:\n'
                              f'All Mean: {all_mean:.3f}\n'
                              f'All Std: {all_std:.3f}\n'
                              f'Benign Mean: {benign_mean:.3f}\n'
                              f'Malicious Mean: {malicious_mean:.3f}\n'
                              f'Difference: {benign_mean - malicious_mean:.3f}')

                if all_mean > np.median(bins):
                    text_x = 0.02
                else:
                    text_x = 0.75
                ax.text(text_x, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                plt.tight_layout()

                plot_path = os.path.join(plot_dir, f"reputation_distribution_round_{self.current_round:04d}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not create reputation distribution plot: {e}")


    def save_model_with_reputation(self, filepath):
        reputations = {client.client_id: client.reputation for client in self.clients}
        new_reputations = {client.client_id: client.new_reputation for client in self.clients}

        save_data = {
            'model_state_dict': self.model.state_dict(),
            'reputations': reputations,
            'new_reputations': new_reputations,
            'current_round': self.current_round,
            'poison_rounds': self.poison_rounds,
            'acc_mal_new_history': self.acc_mal_new_history,
            'lowest_new_reputation_clients_history': self.lowest_new_reputation_clients_history
        }

        torch.save(save_data, filepath)
        print(f"Model and reputations saved to {filepath}")

    def load_model_with_reputation(self, filepath):
        if os.path.exists(filepath):
            save_data = torch.load(filepath, map_location=args.device, weights_only=False)
            self.model.load_state_dict(save_data['model_state_dict'])
            if 'reputations' in save_data:
                reputations = save_data['reputations']
                for client_id, reputation in reputations.items():
                    if client_id < len(self.clients):
                        self.clients[client_id].reputation = reputation
                print(f"Loaded reputations for {len(reputations)} clients")

            if 'new_reputations' in save_data:
                new_reputations = save_data['new_reputations']
                for client_id, new_reputation in new_reputations.items():
                    if client_id < len(self.clients):
                        self.clients[client_id].new_reputation = new_reputation
                print(f"Loaded new reputations for {len(new_reputations)} clients")


            if 'acc_mal_new_history' in save_data:
                self.acc_mal_new_history = save_data['acc_mal_new_history']
                print(f"Loaded ACC_MAL (NEW reputation) history with {len(self.acc_mal_new_history)} rounds")


            if 'lowest_new_reputation_clients_history' in save_data:
                self.lowest_new_reputation_clients_history = save_data['lowest_new_reputation_clients_history']
                print(
                    f"Loaded lowest NEW reputation clients history with {len(self.lowest_new_reputation_clients_history)} rounds")


            if 'current_round' in save_data:
                self.current_round = save_data['current_round']
            if 'poison_rounds' in save_data:
                self.poison_rounds = save_data['poison_rounds']

            print(f"Model and reputations loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found, starting from scratch")

    def validate(self):
        with torch.no_grad():
            test_l, test_acc = test_cv(self.helper.benign_test_data, self.model)
        return test_l, test_acc

    def validate_poison(self):
        with torch.no_grad():
            if args.attack_mode.lower() in ['combine', 'combine2']:
                test_l_acc = []
                for i in range(args.multi_objective_num):
                    test_l, test_acc = test_poison_cv(self.helper, self.helper.poisoned_test_data,
                                                      self.model, adversarial_index=i)
                    test_l_acc.append((test_l, test_acc))
                return test_l_acc
            else:
                test_l, test_acc = test_poison_cv(self.helper, self.helper.poisoned_test_data, self.model)
                return test_l, test_acc