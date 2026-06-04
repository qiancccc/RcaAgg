import copy
import random
import time

import torch

from FL_Backdoor_CV.roles.evaluation import test_poison_cv, test_cv
from configs import args

import torch.optim as optim

class Client:
    def __init__(self, client_id, local_data, local_data_size, malicious=False):
        self.client_id = client_id
        self.local_data = local_data
        self.local_data_size = local_data_size
        self.malicious = malicious
        self.server = None
        self.adversarial_index = None
        self.range_no_id = None
        self.reputation = 0.0
        self.new_reputation = 0.0
        self.new_reputation = random.uniform(0, 0.01)

        self.is_adaptive_adversary = False
        self.adaptive_attack_start_round = args.adaptive_attack_start_round

        self.reputation_history = []
        self.history_window = 5

        self.performance_bonus_active = False
        self.consecutive_positive_count = 0


    def update_reputation(self, credit_score, mean_normalized_score, all_adjustments=None,cluster_size=1,bottom_10_percent_clients=None, bottom_10_30_percent_clients=None):
        normalized_score = max(0.0, min(1.0, credit_score))
        current_adjustment = normalized_score - mean_normalized_score
        penalty_multiplier = 1.0
        if args.penalty:
            if bottom_10_percent_clients is not None and self.client_id in bottom_10_percent_clients:
                penalty_multiplier = 2
            elif bottom_10_30_percent_clients is not None and self.client_id in bottom_10_30_percent_clients:
                penalty_multiplier = 1.5
        current_adjustment *= penalty_multiplier
        ranking_multiplier = 1.0

        if args.reward and cluster_size == 1 and all_adjustments is not None and len(all_adjustments) > 1:
            sorted_adjustments = sorted(all_adjustments, reverse=True)
            current_rank = sorted_adjustments.index(current_adjustment) + 1
            total_clients = len(all_adjustments)
            rank_percentile = current_rank / total_clients
            if rank_percentile <= 0.2:
                ranking_multiplier = 3.0
            elif rank_percentile <= 0.4:
                ranking_multiplier = 2
        current_adjustment *= ranking_multiplier

        self.reputation_history.append(current_adjustment)
        if len(self.reputation_history) > self.history_window:
            self.reputation_history.pop(0)

        if len(self.reputation_history) >= 3:
            historical_adjustment = sum(self.reputation_history) / len(self.reputation_history)
        else:
            historical_adjustment = current_adjustment

        alpha = 0.7
        final_adjustment = alpha * historical_adjustment + (1 - alpha) * current_adjustment
        self.reputation += final_adjustment
        self.reputation = max(-10.0, min(10.0, self.reputation))

    def update_new_reputation(self, weighted_variance_rank_head, weighted_variance_rank_tail):
        for rank, cluster in enumerate(weighted_variance_rank_head, start=1):
            if self.client_id in cluster:
                weight = 1.0 / rank
                self.new_reputation -= weight
                break

        reversed_tail_clusters = list(reversed(weighted_variance_rank_tail))
        for rank, cluster in enumerate(reversed_tail_clusters, start=1):
            if self.client_id in cluster:
                weight = 1.0 / rank
                self.new_reputation += weight
                break
        self.new_reputation = max(-500.0, min(500.0, self.new_reputation))

    def update_nnew_reputation(self, weighted_variance_rank_head, weighted_variance_rank_tail,
                               avg_reputation_rank_head, avg_reputation_rank_tail):
        for cluster in weighted_variance_rank_head:
            if self.client_id in cluster and cluster in avg_reputation_rank_tail:
                self.new_reputation -= 1
                break
        for cluster in weighted_variance_rank_tail:
            if self.client_id in cluster and cluster in avg_reputation_rank_head:
                self.new_reputation += 1
                break
        self.new_reputation = max(-500.0, min(500.0, self.new_reputation))

    def local_train(self, local_model, helper, epoch, criterion=torch.nn.CrossEntropyLoss()):
        local_data = self.local_data
        is_attack_round = self.server.current_round in self.server.poison_rounds

        if self.is_adaptive_adversary:
            current_round = self.server.current_round
            if current_round == self.adaptive_attack_start_round:
                print(f"🔄 Adaptive Adversary {self.client_id} starts attacking at round {current_round}")
            if current_round < self.adaptive_attack_start_round:
                self.malicious = False
            else:
                self.malicious = True

        if self.malicious and is_attack_round:
            if self.range_no_id is None:
                range_no_id = list(range(len(helper.train_dataset)))
                if args.attack_mode.lower() in ['mr', 'dba', 'flip']:
                    for ind, x in enumerate(helper.train_dataset):
                        imge, label = x
                        if label == helper.params['poison_label_swap']:
                            range_no_id.remove(ind)

                    if args.dataset == 'cifar10':
                        if args.attack_mode.lower() == 'mr':
                            for image in helper.params['poison_images_test'] + \
                                         helper.params['poison_images']:
                                if image in range_no_id:
                                    range_no_id.remove(image)

                elif args.attack_mode.lower() == 'combine':
                    target_label = None
                    if self.adversarial_index == 0:
                        target_label = helper.params['poison_label_swaps'][0]
                    elif self.adversarial_index == 1:
                        target_label = helper.params['poison_label_swaps'][1]
                    elif self.adversarial_index == 2:
                        target_label = helper.params['poison_label_swaps'][2]
                    elif self.adversarial_index == 3:
                        target_label = helper.params['poison_label_swap']
                    for ind, x in enumerate(helper.train_dataset):
                        imge, label = x
                        if label == target_label:
                            range_no_id.remove(ind)

                    if args.dataset == 'cifar10':
                        if self.adversarial_index == 3:
                            for image in helper.params['poison_images_test'] + \
                                         helper.params['poison_images']:
                                if image in range_no_id:
                                    range_no_id.remove(image)
                elif args.attack_mode.lower() == 'combine2':
                    target_label = helper.params['poison_label_swaps'][self.adversarial_index]
                    for ind, x in enumerate(helper.train_dataset):
                        imge, label = x
                        if label == target_label:
                            range_no_id.remove(ind)
                elif args.attack_mode.lower() in ['neurotoxin', 'edge_case']:
                    pass
                random.shuffle(range_no_id)
                self.range_no_id = range_no_id
            # === malicious training ===
            current_poison_lr = helper.params['poison_lr']
            poison_optimizer = optim.SGD(local_model.parameters(), lr=current_poison_lr,
                                         momentum=helper.params['poison_momentum'],
                                         weight_decay=helper.params['poison_decay'])
            if args.attack_mode.lower() in ['mr', 'dba', 'flip', 'combine']:
                for internal_epoch in range(1, 1 + helper.params['retrain_poison']):
                    indices = random.sample(self.range_no_id, args.batch_size - args.num_poisoned_samples)
                    if args.alternating_minimization:
                        for x in helper.poisoned_train_data:
                            inputs_p, labels_p = None, None
                            if args.attack_mode.lower() == 'mr':
                                inputs_p, labels_p = helper.get_poison_batch(x)
                            elif args.attack_mode.lower() in ['dba', 'combine']:
                                inputs_p, labels_p = helper.get_poison_batch(x, adversarial_index=self.adversarial_index)
                            elif args.attack_mode.lower() == 'flip':
                                inputs_p, labels_p = x
                                for pos in range(labels_p.size(0)):
                                    labels_p[pos] = helper.params['poison_label_swap']
                            poison_optimizer.zero_grad()
                            output = local_model(inputs_p.to(args.device))
                            loss = criterion(output, labels_p.to(args.device))
                            loss.backward()
                            poison_optimizer.step()
                            break

                        for x in helper.get_train(indices):
                            inputs_c, labels_c = x
                            poison_optimizer.zero_grad()
                            output = local_model(inputs_c.to(args.device))
                            loss = criterion(output, labels_c.to(args.device))
                            loss.backward()
                            poison_optimizer.step()
                            break

                    else:

                        for (x1, x2) in zip(helper.poisoned_train_data, helper.get_train(indices)):
                            inputs_p, labels_p = None, None
                            if args.attack_mode.lower() == 'mr':
                                inputs_p, labels_p = helper.get_poison_batch(x1)
                            elif args.attack_mode.lower() in ['dba', 'combine']:
                                inputs_p, labels_p = helper.get_poison_batch(x1, adversarial_index=self.adversarial_index)
                            elif args.attack_mode.lower() == 'flip':
                                inputs_p, labels_p = x1
                                for pos in range(labels_p.size(0)):
                                    labels_p[pos] = helper.params['poison_label_swap']
                            inputs_c, labels_c = x2
                            if args.attack_mode.lower() == 'flip':
                                for pos in range(labels_c.size(0)):
                                    if labels_c[pos] == 7:
                                        labels_c[pos] = helper.params['poison_label_swap']
                            inputs = torch.cat((inputs_p, inputs_c))
                            labels = torch.cat((labels_p, labels_c))
                            inputs, labels = inputs.to(args.device), labels.to(args.device)
                            poison_optimizer.zero_grad()
                            output = local_model(inputs)
                            loss = criterion(output, labels)
                            loss.backward()
                            poison_optimizer.step()

                            break

                    # === test poison ===
                    if args.show_process:
                        if args.attack_mode.lower() == 'combine':
                            poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model,
                                                                     self.adversarial_index)
                        else:
                            poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model)
                        print(f"Malicious id: {self.client_id}, "
                              f"P o i s o n - N o w ! "
                              f"Epoch: {internal_epoch}, "
                              f"Local poison accuracy: {poison_acc: .4f}, "
                              f"Local poison loss: {poison_loss: .4f}.")
                    else:
                        if internal_epoch % helper.params['retrain_poison'] == 0:
                            if args.attack_mode.lower() == 'combine':
                                poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model, self.adversarial_index)
                            else:
                                poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model)
                            print(f"Malicious id: {self.client_id}, "
                                  f"P o i s o n - N o w ! "
                                  f"Epoch: {internal_epoch}, "
                                  f"Local poison accuracy: {poison_acc: .4f}, "
                                  f"Local poison loss: {poison_loss: .4f}.")
            elif args.attack_mode.lower() == 'combine2':
                for internal_epoch in range(1, 1 + helper.params['retrain_poison']):
                    indices = random.sample(self.range_no_id, args.batch_size - args.num_poisoned_samples)
                    if args.alternating_minimization:

                        for x in helper.poisoned_train_data:
                            inputs_p, labels_p = helper.get_poison_batch(x, adversarial_index=self.adversarial_index)
                            poison_optimizer.zero_grad()
                            output = local_model(inputs_p.to(args.device))
                            loss = criterion(output, labels_p.to(args.device))
                            loss.backward()
                            poison_optimizer.step()
                            break

                        for x in helper.get_train(indices):
                            inputs_c, labels_c = x
                            poison_optimizer.zero_grad()
                            output = local_model(inputs_c.to(args.device))
                            loss = criterion(output, labels_c.to(args.device))
                            loss.backward()
                            poison_optimizer.step()
                            break
                    else:

                        for (x1, x2) in zip(helper.poisoned_train_data, helper.get_train(indices)):
                            inputs_p, labels_p = helper.get_poison_batch(x1, adversarial_index=self.adversarial_index)
                            inputs_c, labels_c = x2
                            inputs = torch.cat((inputs_p, inputs_c))
                            labels = torch.cat((labels_p, labels_c))
                            inputs, labels = inputs.to(args.device), labels.to(args.device)
                            poison_optimizer.zero_grad()
                            output = local_model(inputs)
                            loss = criterion(output, labels)
                            loss.backward()
                            poison_optimizer.step()
                            break

                    if args.show_process or (internal_epoch % helper.params['retrain_poison'] == 0):
                                poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model,
                                                                         adversarial_index=self.adversarial_index)
                                print(f"Malicious id: {self.client_id}, Poison Now! Epoch: {internal_epoch}, "
                                      f"Local poison accuracy: {poison_acc:.4f}, loss: {poison_loss:.4f}")

            elif args.attack_mode.lower() in ['edge_case', 'neurotoxin']:
                # === get gradient mask use global model and clean data ===
                mask_grad_list = None
                if args.attack_mode.lower() == 'neurotoxin':
                    assert helper.params['gradmask_ratio'] != 1
                    num_clean_data = 150
                    subset_data_chunks = random.sample(helper.params['participant_clean_data'], num_clean_data)
                    sampled_data = [helper.benign_train_data[pos] for pos in subset_data_chunks]
                    mask_grad_list = helper.grad_mask_cv(helper, local_model, sampled_data, criterion,
                                                         ratio=helper.params['gradmask_ratio'])
                reduced_epochs = max(1, helper.params['retrain_poison'] // 2)
                for internal_epoch in range(1, 1 + reduced_epochs):
                    if not args.alternating_minimization:

                        for x in helper.poisoned_train_data:
                            inputs_p, labels_p = x

                            for pos in range(labels_p.size(0)):
                                labels_p[pos] = helper.params['poison_label_swap']
                            inputs, labels = inputs_p.to(args.device), labels_p.to(args.device)
                            poison_optimizer.zero_grad()
                            output = local_model(inputs)
                            loss = criterion(output, labels)
                            loss.backward()

                            if args.attack_mode.lower() == 'neurotoxin':
                                mask_grad_list_copy = iter(mask_grad_list)
                                for name, parms in local_model.named_parameters():
                                    if parms.requires_grad:
                                        parms.grad = parms.grad * next(mask_grad_list_copy)
                            poison_optimizer.step()
                            break

                        indices = random.sample(self.range_no_id, args.batch_size - args.num_poisoned_samples)
                        for x in helper.get_train(indices):
                            inputs_c, labels_c = x
                            inputs, labels = inputs_c.to(args.device), labels_c.to(args.device)
                            poison_optimizer.zero_grad()
                            output = local_model(inputs)
                            loss = criterion(output, labels)
                            loss.backward()
                            poison_optimizer.step()
                            break
                    else:
                        indices = random.sample(self.range_no_id, args.batch_size - args.num_poisoned_samples)
                        for (x1, x2) in zip(helper.poisoned_train_data, helper.get_train(indices)):
                            inputs_p, labels_p = x1
                            inputs_c, labels_c = x2
                            inputs = torch.cat((inputs_p, inputs_c))
                            for pos in range(labels_p.size(0)):
                                labels_p[pos] = helper.params['poison_label_swap']
                            labels = torch.cat((labels_p, labels_c))
                            inputs, labels = inputs.to(args.device), labels.to(args.device)
                            poison_optimizer.zero_grad()
                            output = local_model(inputs)
                            loss = criterion(output, labels)
                            loss.backward()
                            if args.attack_mode.lower() == 'neurotoxin':
                                mask_grad_list_copy = iter(mask_grad_list)
                                for name, parms in local_model.named_parameters():
                                    if parms.requires_grad:
                                        parms.grad = parms.grad * next(mask_grad_list_copy)
                            poison_optimizer.step()

                    if args.show_process:
                        poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model)
                        print(f"Malicious id: {self.client_id}, "
                              f"P o i s o n - N o w ! "
                              f"Epoch: {internal_epoch}, "
                              f"Local poison accuracy: {poison_acc: .4f}, "
                              f"Local poison loss: {poison_loss: .4f}.")
                    else:
                        if internal_epoch % reduced_epochs == 0:
                            poison_loss, poison_acc = test_poison_cv(helper, helper.poisoned_test_data, local_model)
                            print(f"Malicious id: {self.client_id}, "
                                  f"P o i s o n - N o w ! "
                                  f"Epoch: {internal_epoch}, "
                                  f"Local poison accuracy: {poison_acc: .4f}, "
                                  f"Local poison loss: {poison_loss: .4f}.")

            # === malicious test ===
            test_loss, test_acc = test_cv(helper.benign_test_data, local_model)
            print(f"Malicious id: {self.client_id}, "
                  f"Test accuracy: {test_acc: .4f}, "
                  f"Test loss: {test_loss: .4f}.")
        else:
            if args.aggregation_rule.lower() in ['rcaagg', 'avg', 'fltrust','flame', 'foolsgold', 'rlr']:
                if epoch > 300:
                    lr = args.local_lr * args.local_lr_decay ** ((epoch - 300) // args.decay_step)
                else:
                    lr = args.local_lr
            optimizer = optim.SGD(local_model.parameters(), lr=lr,
                                  momentum=helper.params['momentum'],
                                  weight_decay=helper.params['decay'])
            epochs = helper.params['retrain_no_times']

            # === local training ===
            for _ in range(epochs):
                for inputs, labels in local_data:
                    inputs, labels = inputs.to(args.device), labels.to(args.device)
                    optimizer.zero_grad()
                    loss = criterion(local_model(inputs), labels)
                    loss.backward()
                    optimizer.step()
        return local_model
