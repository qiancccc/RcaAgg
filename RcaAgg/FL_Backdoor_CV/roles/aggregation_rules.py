import math
from collections import defaultdict

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
import torch.nn.functional as F
import hdbscan
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh as largest_eigh
from sklearn.cluster import DBSCAN, KMeans

from configs import args
from sklearn.metrics import pairwise_distances


def safe_eigsh(matrix, k=1):
    import numpy as np
    from scipy.sparse.linalg import eigsh
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    matrix = matrix.astype(np.float64)
    n = matrix.shape[0]
    if n != matrix.shape[1]:
        min_dim = min(matrix.shape[0], matrix.shape[1])
        matrix = matrix[:min_dim, :min_dim]
        n = min_dim
    if n <= k:
        eigenvalues = np.ones(k)
        eigenvectors = np.eye(n, k)
        return eigenvalues, eigenvectors
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        matrix = (matrix + matrix.T) / 2.0
    eps = 1e-10
    regularized_matrix = matrix + eps * np.eye(n)
    try:
        eigenvalues, eigenvectors = eigsh(
            regularized_matrix,
            k=min(k, n - 1),
            which='LM',
            maxiter=1000,
            tol=1e-6
        )
        return eigenvalues, eigenvectors
    except Exception as e:
        print(e)

def update_participation(conspiracy_graph, client_id):
    if client_id not in conspiracy_graph.client_participation_rounds:
        conspiracy_graph.client_participation_rounds[client_id] = 0
        conspiracy_graph.client_malicious_rounds[client_id] = 0
    conspiracy_graph.client_participation_rounds[client_id] += 1

def update_conspiracy_edges(conspiracy_graph, cluster):
    for client_id in cluster:
        if client_id not in conspiracy_graph.client_malicious_rounds:
            conspiracy_graph.client_malicious_rounds[client_id] = 0
        conspiracy_graph.client_malicious_rounds[client_id] += 1
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            client_u, client_v = sorted([cluster[i], cluster[j]])
            edge_key = f"{client_u}-{client_v}"
            if edge_key not in conspiracy_graph.conspiracy_edges:
                conspiracy_graph.conspiracy_edges[edge_key] = 1
            else:
                conspiracy_graph.conspiracy_edges[edge_key] += 1

def calculate_suspect_scores(conspiracy_graph, participants, credit_scores, global_sim_idxs,
                              reputation_dict, client_reputations):
    if conspiracy_graph is None or global_sim_idxs is None or participants is None:
        return {}
    singleton_clients = identify_singleton_clients(global_sim_idxs, participants)
    suspect_scores = {}
    for client_id in range(args.participant_population):
        suspect_scores[client_id] = 0.0
    for client_id in singleton_clients:
        client_malicious_rounds = conspiracy_graph.client_malicious_rounds.get(client_id, 0)
        if client_malicious_rounds < args.min_conspiracy_rounds:
            continue
        edge_weight_sum = 0
        for edge_key, weight in conspiracy_graph.conspiracy_edges.items():
            client_u, client_v = map(int, edge_key.split('-'))
            if client_id == client_u or client_id == client_v:
                edge_weight_sum += weight
        participation_rounds = conspiracy_graph.client_participation_rounds.get(client_id, 1)
        conspiracy_frequency = edge_weight_sum / participation_rounds if participation_rounds > 0 else 0
        suspect_scores[client_id] = conspiracy_frequency
    return suspect_scores

def _log_suspect_clients(suspect_scores, conspiracy_graph):
    if not suspect_scores:
        return
    all_high_suspect = []
    punishable_high_suspect = []

    for client_id, score in suspect_scores.items():
        if score >= args.suspect_threshold:
            all_high_suspect.append((client_id, score))
            client_malicious_rounds = conspiracy_graph.client_malicious_rounds.get(client_id, 0)
            if client_malicious_rounds >= args.min_conspiracy_rounds:
                punishable_high_suspect.append((client_id, score, client_malicious_rounds))
    print(f"score≥{args.suspect_threshold}: {len(all_high_suspect)}")
    print(
        f"（score≥{args.suspect_threshold} and min_round≥{args.min_conspiracy_rounds}）num: {len(punishable_high_suspect)}")
    if punishable_high_suspect:
        print(f"Punish high-risk client machines:")
        for client_id, score, malicious_rounds in punishable_high_suspect:
            client_type = "Maliciousness" if client_id < args.number_of_adversaries else "Benign"
            print(f"  client {client_id} ({client_type}): score={score:.3f}, m_round={malicious_rounds}")


def identify_singleton_clients(global_sim_idxs, participant_indices):
    multi_cluster_clients = set()
    for cluster in global_sim_idxs:
        for client_id in cluster:
            multi_cluster_clients.add(client_id)
    all_participants = set(participant_indices)
    singleton_clients = all_participants - multi_cluster_clients
    return singleton_clients

def update_collaborative_security(conspiracy_graph,  participants, bad_clusters,
                                  credit_scores, global_sim_idxs, reputation_dict, client_reputations):
    if not args.collaborative_security or not conspiracy_graph:
        return conspiracy_graph, {}
    for client_id in participants:
        update_participation(conspiracy_graph, client_id)
    for cluster in bad_clusters:
        update_conspiracy_edges(conspiracy_graph, cluster)
    suspect_scores = calculate_suspect_scores(
        conspiracy_graph=conspiracy_graph,
        participants=participants,
        credit_scores=credit_scores,
        global_sim_idxs=global_sim_idxs,
        reputation_dict=reputation_dict,
        client_reputations=client_reputations
    )
    _log_suspect_clients(suspect_scores, conspiracy_graph)
    return conspiracy_graph, suspect_scores


def adaptive_dbscan(indicative_layer_updates):
    data = indicative_layer_updates.cpu().numpy()
    initial_eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_overall_labels = None
    best_overall_eps = initial_eps_values[0]
    best_overall_score = float('inf')
    target_min, target_max = 12, 15

    for start_eps in initial_eps_values:
        result, eps, score, n_clusters = run_single_search(data, start_eps, target_min, target_max)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_labels = result.labels_
            best_overall_eps = eps
        if score == 0:
            break
    class ClusterResult:
        def __init__(self, labels):
            self.labels_ = labels
    final_result = ClusterResult(best_overall_labels)
    return final_result, best_overall_eps

def run_single_search(data, start_eps, target_min, target_max):
    current_eps = start_eps
    eps_min, eps_max = 0.05, 0.8
    max_iterations = 10
    best_labels = None
    best_eps = current_eps
    best_score = float('inf')
    best_n_clusters = 0
    last_score = float('inf')
    adjustment_direction = None
    for iteration in range(max_iterations):
        dbscan = DBSCAN(eps=current_eps, min_samples=1, metric='cosine')
        labels = dbscan.fit_predict(data)
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        valid_clusters = unique_labels[counts >= 2]
        n_valid_clusters = len(valid_clusters)

        if n_valid_clusters < target_min:
            score = target_min - n_valid_clusters
        elif n_valid_clusters > target_max:
            score = n_valid_clusters - target_max
        else:
            score = 0

        if score < best_score:
            best_score = score
            best_labels = labels
            best_eps = current_eps
            best_n_clusters = n_valid_clusters

        if score == 0:
            break

        if iteration == 0:
            if n_valid_clusters < target_min:
                adjustment_direction = 'decrease'
            else:
                adjustment_direction = 'increase'
        elif score > last_score:
            # 结果变差，反向
            if adjustment_direction == 'decrease':
                adjustment_direction = 'increase'
            else:
                adjustment_direction = 'decrease'
        last_score = score

        if adjustment_direction == 'decrease':
            eps_max = current_eps
            current_eps = max(eps_min, (current_eps + eps_min) / 2)
        else:
            eps_min = current_eps
            current_eps = min(eps_max, (current_eps + eps_max) / 2)
    class ClusterResult:
        def __init__(self, labels):
            self.labels_ = labels
    return ClusterResult(best_labels), best_eps, best_score, best_n_clusters


def avg(model_updates):
    global_update = dict()
    for name, data in model_updates.items():
        global_update[name] = 1 / args.participant_sample_size * model_updates[name].sum(dim=0, keepdim=True)
    return global_update

def rcaagg(model_updates, current_round=0,client_reputations=None, participant_indices=None,reputation_dict=None, client_new_reputations=None,stop_reputation_update=False, conspiracy_graph=None, all_suspect_scores=None):
    client_cluster_sizes = {}
    # === construct a distance matrix ===
    keys = list(model_updates.keys())
    indicative_layer_updates = F.normalize(model_updates[keys[-2]])
    K = len(indicative_layer_updates)
    distance_matrix_2 = pairwise_distances(indicative_layer_updates.cpu().numpy(), metric='euclidean') ** 2
    distance_matrix_2 = torch.tensor(distance_matrix_2, dtype=torch.float32)
    partition, adjusted_eps = adaptive_dbscan(indicative_layer_updates)
    labels = partition.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    valid_clusters = unique_labels[counts >= 2]
    n_valid_clusters = len(valid_clusters)
    print(f"DBSCAN: eps={adjusted_eps:.3f}, num={n_valid_clusters}")

    clusters = dict()
    for i, clu in enumerate(partition.labels_):
        clu_key = int(clu)
        if clu_key in clusters:
            clusters[clu_key].append(i)
        else:
            clusters[clu_key] = [i]
        local_to_global_map = {}
        if participant_indices is not None:
            for local_idx, global_id in enumerate(participant_indices):
                local_to_global_map[local_idx] = global_id
            client_cluster_sizes = {}
            for local_idx in range(len(partition.labels_)):
                cluster_label = partition.labels_[local_idx]
                cluster_label_int = int(cluster_label)
                if cluster_label_int in clusters:
                    cluster_size = len(clusters[cluster_label_int])
                else:
                    cluster_size = 1

                if participant_indices is not None:
                    global_client_id = participant_indices[local_idx]
                    client_cluster_sizes[global_client_id] = cluster_size
                else:

                    client_cluster_sizes[local_idx] = cluster_size

    # === find updates with similar directional contribution ===
    sim_idxs = list()
    for clu in clusters.values():
        if len(clu) > 1:
            sim_idxs.append(clu)

    sim_idxs.sort(key=lambda x: x[0])
    global_sim_idxs = []
    if participant_indices is not None:
        for local_group in sim_idxs:
            global_group = [participant_indices[i] for i in local_group]  # 使用participant_indices参数
            global_sim_idxs.append(sorted(global_group))
    else:
        global_sim_idxs = sim_idxs

    # === find the master index and remove the other indices within each cluster ===
    remove_idxs = []
    for idxs in sim_idxs:
        idxs = list(idxs)
        idxs.sort()
        if remove_idxs:
            remove_idxs.extend(idxs[1:])
        else:
            remove_idxs = idxs[1:]
    reserve_idxs = list(set(list(range(K))) - set(remove_idxs))
    avg_reputation_rank_head = []
    avg_reputation_rank_tail = []
    # 计算每个簇的平均信誉值
    cluster_avg_reputations = []
    for cluster in global_sim_idxs:
        if reputation_dict is not None:
          cluster_reputations = [reputation_dict[client_id] for client_id in cluster]
          avg_reputation = sum(cluster_reputations) / len(cluster_reputations)
          cluster_avg_reputations.append(avg_reputation)
        else:
          cluster_avg_reputations.append(0.0)
    print(f'Clients with similar directional contribution:')
    print(f'  Total # Clusters: {max(partition.labels_) + 1}')
    if cluster_avg_reputations and global_sim_idxs:
        clusters_with_avg_rep = list(zip(global_sim_idxs, cluster_avg_reputations))
        clusters_sorted_by_avg_rep = sorted(clusters_with_avg_rep, key=lambda x: x[1], reverse=True)
        total_clusters = len(clusters_sorted_by_avg_rep)
        head_count = max(1, int(total_clusters * 0.3))
        tail_count = max(1, int(total_clusters * 0.3))
        head_clusters = clusters_sorted_by_avg_rep[:head_count]
        tail_clusters = clusters_sorted_by_avg_rep[-tail_count:]
        for cluster_info in head_clusters:
            cluster = cluster_info[0]
            avg_reputation_rank_head.append(cluster)
        for cluster_info in tail_clusters:
            cluster = cluster_info[0]
            avg_reputation_rank_tail.append(cluster)
    weighted_variance_rank_head = []
    weighted_variance_rank_tail = []

    if cluster_avg_reputations:

        cluster_variances = []
        cluster_weighted_variances = []
        all_reputations = []
        for cluster in global_sim_idxs:
            if reputation_dict is not None:
                cluster_reputations = [reputation_dict[client_id] for client_id in cluster]
                all_reputations.extend(cluster_reputations)
        grand_mean = np.mean(all_reputations) if all_reputations else 0.0
        for cluster in global_sim_idxs:
            if reputation_dict is not None:
                cluster_reputations = [reputation_dict[client_id] for client_id in cluster]
                n = len(cluster_reputations)
                if n > 1:

                    mean_rep = np.mean(cluster_reputations)
                    sum_sq_diff = sum((x - mean_rep) ** 2 for x in cluster_reputations)
                    variance = sum_sq_diff / (n - 1)
                else:
                    variance = 0.0
                cluster_variances.append(variance)
                weighted_variance = variance * (1 / (n - 1)) if n > 1 else 0.0
                cluster_weighted_variances.append(weighted_variance)
            else:
                cluster_variances.append(0.0)
                cluster_weighted_variances.append(0.0)

        ranked_clusters_by_reputation = sorted(
            list(zip(global_sim_idxs, cluster_avg_reputations, cluster_variances, cluster_weighted_variances)),
            key=lambda x: x[1],
            reverse=True
        )

        ranked_clusters_by_variance = sorted(
            list(zip(global_sim_idxs, cluster_avg_reputations, cluster_variances, cluster_weighted_variances)),
            key=lambda x: x[2],
            reverse=False
        )

        ranked_clusters_by_weighted_variance = sorted(
            list(zip(global_sim_idxs, cluster_avg_reputations, cluster_variances, cluster_weighted_variances)),
            key=lambda x: x[3],
            reverse=False
        )

        total_clusters = len(ranked_clusters_by_weighted_variance)
        head_count = max(1, int(total_clusters * 0.3))
        tail_count = max(1, int(total_clusters * 0.3))
        head_clusters = ranked_clusters_by_weighted_variance[:head_count]
        tail_clusters = ranked_clusters_by_weighted_variance[-tail_count:]
        for cluster_info in head_clusters:
            cluster = cluster_info[0]
            weighted_variance_rank_head.append(cluster)
        for cluster_info in tail_clusters:
            cluster = cluster_info[0]
            weighted_variance_rank_tail.append(cluster)

        reputation_rank_dict = {}
        variance_rank_dict = {}
        weighted_variance_rank_dict = {}
        for rank, (cluster, _, _, _) in enumerate(ranked_clusters_by_reputation, 1):
            cluster_key = tuple(cluster)
            reputation_rank_dict[cluster_key] = rank
        for rank, (cluster, _, _, _) in enumerate(ranked_clusters_by_variance, 1):
            cluster_key = tuple(cluster)
            variance_rank_dict[cluster_key] = rank
        for rank, (cluster, _, _, _) in enumerate(ranked_clusters_by_weighted_variance, 1):
            cluster_key = tuple(cluster)
            weighted_variance_rank_dict[cluster_key] = rank

        for cluster, avg_reputation, variance, weighted_variance in zip(global_sim_idxs, cluster_avg_reputations,
                                                                        cluster_variances, cluster_weighted_variances):
            reputation_rank = reputation_rank_dict[tuple(cluster)]
            variance_rank = variance_rank_dict[tuple(cluster)]
            weighted_variance_rank = weighted_variance_rank_dict[tuple(cluster)]
            def format_precision(num):
                if num == 0:
                    return "0.000000"
                elif abs(num) < 1e-6:
                    return f"{num:.2e}"
                else:
                    return f"{num:.6f}"
            avg_reputation_str = format_precision(avg_reputation)
            variance_str = format_precision(variance)
            weighted_variance_str = format_precision(weighted_variance)
            print(
                f'  Cluster {cluster} -> Avg Rep: {avg_reputation_str}, Rep Rank: {reputation_rank}/{len(cluster_avg_reputations)}, '
                f'Var: {variance_str}, Var Rank: {variance_rank}/{len(cluster_variances)}, '
                f'Weighted Var: {weighted_variance_str}, WVar Rank: {weighted_variance_rank}/{len(cluster_weighted_variances)}')
    else:
        print('  No clusters found')

    if all_suspect_scores is not None and global_sim_idxs:
        cluster_suspect_sums = []
        for cluster in global_sim_idxs:
            suspect_sum = sum(all_suspect_scores.get(client_id, 0.0) for client_id in cluster)
            cluster_suspect_sums.append(suspect_sum)
        min_avg_rep = min(cluster_avg_reputations) if cluster_avg_reputations else -1.0
        adjusted_avg_reputations = []
        for avg_rep, suspect_sum, cluster in zip(cluster_avg_reputations, cluster_suspect_sums, global_sim_idxs):
            if avg_rep < 0:
                cluster_size = len(cluster)
                adjustment_factor = suspect_sum / cluster_size if cluster_size > 0 else 0
                adjusted_rep = max(min_avg_rep, avg_rep * adjustment_factor)
            else:
                adjusted_rep = avg_rep
            adjusted_avg_reputations.append(adjusted_rep)
    else:
        adjusted_avg_reputations = cluster_avg_reputations.copy() if cluster_avg_reputations else []

    bad_clusters = []
    if args.remove_bad_clusters and args.num_bad_clusters > 0 and reputation_rank_dict:
        bad_clusters_count = min(args.num_bad_clusters, len(global_sim_idxs))
        if adjusted_avg_reputations:
            sorted_avg_reps = sorted(adjusted_avg_reputations)
            negative_reps = [rep for rep in sorted_avg_reps if rep < 0]
            positive_reps = [rep for rep in sorted_avg_reps if rep >= 0]

            if len(negative_reps) > 1:
                if not positive_reps:
                    bad_clusters_count = len(adjusted_avg_reputations)
                    print(f"num: {bad_clusters_count}")
                else:
                    detection_reps = negative_reps.copy()
                    detection_reps.append(positive_reps[0])
                    if len(detection_reps) >= 3:
                        gaps = [detection_reps[i + 1] - detection_reps[i] for i in range(len(detection_reps) - 1)]
                        if len(gaps) >= 2:
                            sorted_gaps = sorted(gaps, reverse=True)
                            max_gap = sorted_gaps[0]
                            second_max_gap = sorted_gaps[1]
                            avg_gap = sum(gaps) / len(gaps)
                            has_breakpoint = (max_gap > 1.5 * second_max_gap or max_gap > 1.5 * avg_gap)
                            if has_breakpoint:
                                max_gap_idx = gaps.index(max_gap)
                                dynamic_num_bad = max_gap_idx + 1
                                print(f"bad_num: {dynamic_num_bad}")
                                bad_clusters_count = min(dynamic_num_bad, len(global_sim_idxs))
                            else:
                                print(f"default")
                        else:
                            print("default")
                    else:
                        if len(detection_reps) == 2 and detection_reps[0] < 0 and detection_reps[1] < 0:
                            bad_clusters_count = 2
                            print(f"bad_num2")
                        else:
                            print(f"default")
            else:
                print(f"default")

        if adjusted_avg_reputations:
            clusters_with_adjusted_rep = list(zip(global_sim_idxs, adjusted_avg_reputations))
            clusters_sorted_for_removal = sorted(clusters_with_adjusted_rep, key=lambda x: x[1])
            bad_clusters = [list(cluster) for cluster, _ in clusters_sorted_for_removal[:bad_clusters_count]]
            for rank, (cluster, adjusted_rep) in enumerate(clusters_sorted_for_removal[:bad_clusters_count], 1):
                print(
                    f"  rank{rank}/{len(global_sim_idxs)}: cluster (client: {list(cluster)})，avg_reputation: {adjusted_rep:.6f}")
        else:
            clusters_with_original_rep = list(zip(global_sim_idxs, cluster_avg_reputations))
            clusters_sorted_for_removal = sorted(clusters_with_original_rep, key=lambda x: x[1])  # 升序，值低的在前
            bad_clusters = [list(cluster) for cluster, _ in clusters_sorted_for_removal[:bad_clusters_count]]

            print(f"=== low: {bad_clusters_count} ===")

        if not hasattr(args, 'bad_clusters_removed'):
            args.bad_clusters_removed = 0
        if not hasattr(args, 'malicious_bad_clusters_removed'):
            args.malicious_bad_clusters_removed = 0

        malicious_clusters_this_round = 0
        for cluster, _ in clusters_sorted_for_removal[:bad_clusters_count]:
            if any(client_id < args.number_of_adversaries for client_id in cluster):
                malicious_clusters_this_round += 1
        args.bad_clusters_removed += bad_clusters_count
        args.malicious_bad_clusters_removed += malicious_clusters_this_round
        if args.bad_clusters_removed > 0:
            accuracy = args.malicious_bad_clusters_removed / args.bad_clusters_removed
            print(f"global: {args.malicious_bad_clusters_removed}/{args.bad_clusters_removed} = {accuracy:.2%}")

    # === partial aggregation. The following process intends to assign a weight to each update ===
    weights = torch.ones(K)
    sims = dict()
    if sim_idxs:
        if stop_reputation_update and args.Aggregation_differentiation:
            cluster_avg_new_reputations = []
            for cluster in global_sim_idxs:
                if client_new_reputations is not None:
                    cluster_new_reps = [client_new_reputations[client_id] for client_id in cluster]
                    avg_new_rep = sum(cluster_new_reps) / len(cluster_new_reps)
                    cluster_avg_new_reputations.append(avg_new_rep)
                else:
                    cluster_avg_new_reputations.append(0.0)
            clusters_with_avg_new_rep = list(zip(global_sim_idxs, cluster_avg_new_reputations))
            clusters_sorted_by_new_rep = sorted(clusters_with_avg_new_rep, key=lambda x: x[1])
            lowest_mal_clusters = clusters_sorted_by_new_rep[:args.mal]
            lowest_mal_cluster_ids = [tuple(cluster) for cluster, _ in lowest_mal_clusters]
            for sim_idx in sim_idxs:
                sim_idx = list(sim_idx)
                sim_idx.sort()
                global_cluster = [participant_indices[idx] for idx in
                                  sim_idx] if participant_indices is not None else sim_idx
                global_cluster_tuple = tuple(sorted(global_cluster))
                if global_cluster_tuple in lowest_mal_cluster_ids:
                    sub_distance_matrix = distance_matrix_2[sim_idx, :][:, sim_idx]
                    sub_weights = sub_distance_matrix.sum(axis=0)
                    sub_weights = sub_weights / sub_weights.sum()
                    for i, idx in enumerate(sim_idx):
                        sims[idx] = sub_weights[i]
                else:
                    cluster_size = len(sim_idx)
                    avg_weight = 1.0 / cluster_size
                    for idx in sim_idx:
                        sims[idx] = avg_weight
        elif not stop_reputation_update and args.Aggregation_differentiation:
            total_clusters = len(global_sim_idxs)
            head_cluster_count = max(1, int(total_clusters * 0.3))
            if 'cluster_weighted_variances' not in locals() or not cluster_weighted_variances:
                for sim_idx in sim_idxs:
                    sim_idx = list(sim_idx)
                    sim_idx.sort()
                    sub_distance_matrix = distance_matrix_2[sim_idx, :][:, sim_idx]
                    sub_weights = sub_distance_matrix.sum(axis=0)
                    sub_weights = sub_weights / sub_weights.sum()
                    for i, idx in enumerate(sim_idx):
                        sims[idx] = sub_weights[i]
            else:
                clusters_with_weighted_var = list(zip(global_sim_idxs, cluster_weighted_variances))
                clusters_sorted_by_weighted_var = sorted(
                    clusters_with_weighted_var,
                    key=lambda x: x[1],
                    reverse=False
                )

                head_clusters = clusters_sorted_by_weighted_var[:head_cluster_count]
                head_cluster_ids = [tuple(cluster) for cluster, _ in head_clusters]

                for sim_idx in sim_idxs:
                    sim_idx = list(sim_idx)
                    sim_idx.sort()
                    global_cluster = [participant_indices[idx] for idx in
                                      sim_idx] if participant_indices is not None else sim_idx
                    global_cluster_tuple = tuple(sorted(global_cluster))
                    is_head_cluster = global_cluster_tuple in head_cluster_ids
                    if is_head_cluster:
                        sub_distance_matrix = distance_matrix_2[sim_idx, :][:, sim_idx]
                        sub_weights = sub_distance_matrix.sum(axis=0)
                        sub_weights = sub_weights / sub_weights.sum()
                        for i, idx in enumerate(sim_idx):
                            sims[idx] = sub_weights[i]
                    else:
                        cluster_size = len(sim_idx)
                        avg_weight = 1.0 / cluster_size
                        for idx in sim_idx:
                            sims[idx] = avg_weight
        else:
            for sim_idx in sim_idxs:
                sim_idx = list(sim_idx)
                sim_idx.sort()
                sub_distance_matrix = distance_matrix_2[sim_idx, :][:, sim_idx]
                sub_weights = sub_distance_matrix.sum(axis=0)
                sub_weights = sub_weights / sub_weights.sum()
                for i, idx in enumerate(sim_idx):
                    sims[idx] = sub_weights[i]
        for i, s in sims.items():
            weights[i] = s

    if all_suspect_scores is not None and global_sim_idxs:
        print("=== Cluster Suspect Sums ===")
        for idx, cluster in enumerate(global_sim_idxs):
            suspect_sum = sum(all_suspect_scores.get(client_id, 0) for client_id in cluster)
            print(f"Cluster {idx + 1}: {cluster} -> Suspect Sum = {suspect_sum:.4f}")

    credit_scores = None
    if client_reputations is not None:
        credit_scores = [0.0] * K
    if client_reputations is not None:
        accumulated_credit_scores = [0.0] * K
    global_update = defaultdict()
    suspect_scores = {}
    if args.collaborative_security and args.remove_bad_clusters:
        conspiracy_graph, suspect_scores = update_collaborative_security(
            conspiracy_graph=conspiracy_graph,
            participants=participant_indices,
            bad_clusters=bad_clusters,
            credit_scores=credit_scores,
            global_sim_idxs=global_sim_idxs,
            client_reputations=client_reputations,
            reputation_dict=reputation_dict
        )
    for name, layer_updates in model_updates.items():
        if 'num_batches_tracked' in name:
            if args.is_poison:
                if args.number_of_adversaries < len(layer_updates):
                    global_update[name] = torch.sum(layer_updates[args.number_of_adversaries:]) / len(
                        layer_updates[args.number_of_adversaries:])
                else:
                    global_update[name] = torch.sum(layer_updates) / len(layer_updates)
            else:
                global_update[name] = torch.sum(layer_updates) / len(layer_updates)
        else:
            # === normalization norm ===
            local_norms = np.array([torch.norm(layer_updates[i]).cpu().numpy() for i in range(K)]).reshape(-1, 1)
            if args.is_poison:
                kmeans = KMeans(n_clusters=2, n_init='auto').fit(local_norms)
                clusters = dict()
                for i, clu in enumerate(kmeans.labels_):
                    if clu in clusters:
                        clusters[clu] += [i]
                    else:
                        clusters[clu] = [i]
                local_norms_0 = local_norms[clusters[0]]
                local_norms_1 = local_norms[clusters[1]]
                if np.median(local_norms_0) > np.median(local_norms_1):
                    normalize_norm = np.median(local_norms_1)
                else:
                    normalize_norm = np.median(local_norms_0)
            else:
                normalize_norm = np.median(local_norms)

            # === plausible clean ingredient ===
            origin_directions = F.normalize(layer_updates)
            origin_directions = weights.view(-1, 1).to(args.device) * origin_directions

            first_idxs = []
            if sim_idxs:
                for idxs in sim_idxs:
                    idxs = list(idxs)
                    idxs.sort()
                    first_idx = idxs[0]
                    first_idxs.append(first_idx)
                    for idx in idxs[1:]:
                        origin_directions[first_idx] = origin_directions[first_idx] + origin_directions[idx]
                    origin_directions[first_idx] /= torch.norm(origin_directions[first_idx])
                origin_directions = origin_directions[reserve_idxs]

            # === extract plausible clean ingredient ===
            N = origin_directions.size(0)
            X = torch.matmul(origin_directions, origin_directions.T)
            evals_large, evecs_large = safe_eigsh(X.detach().cpu().numpy(), k=1)
            evals_large = torch.tensor(evals_large[-1], dtype=torch.float32).to(args.device)
            evecs_large = torch.tensor(evecs_large[:, -1], dtype=torch.float32).to(args.device)
            principal_direction = torch.matmul(evecs_large.view(1, -1), origin_directions).T / torch.sqrt(
                evals_large)

            # === reweight partial aggregated model udpates ===
            new_weights = torch.pow(torch.matmul(principal_direction.view(1, -1), origin_directions.T), 2)
            new_weights = new_weights / new_weights.sum()
            if credit_scores is not None and reserve_idxs is not None:
                new_weights_flat = new_weights.view(-1)
                cluster_scores = {}
                cluster_sizes = {}
                for cluster_idx, sim_group in enumerate(sim_idxs):
                    cluster_sizes[cluster_idx] = len(sim_group)
                    cluster_scores[cluster_idx] = 0.0
                for cluster_idx, sim_group in enumerate(sim_idxs):
                    for client_idx in sim_group:
                        if client_idx in reserve_idxs:
                            pos_in_reserve = reserve_idxs.index(client_idx)
                            if pos_in_reserve < len(new_weights_flat):
                                cluster_scores[cluster_idx] += new_weights_flat[pos_in_reserve].item()
                for cluster_idx, sim_group in enumerate(sim_idxs):
                    if cluster_sizes[cluster_idx] > 0:
                        avg_score = cluster_scores[cluster_idx] / cluster_sizes[cluster_idx]
                        for client_idx in sim_group:
                            credit_scores[client_idx] = avg_score
                for i, client_idx in enumerate(reserve_idxs):
                    client_in_any_cluster = False
                    for sim_group in sim_idxs:
                        if client_idx in sim_group:
                            client_in_any_cluster = True
                            break
                    if not client_in_any_cluster and i < len(new_weights_flat):
                        credit_scores[client_idx] = new_weights_flat[i].item()
            for i in range(K):
                accumulated_credit_scores[i] += credit_scores[i]

            origin_directions += torch.normal(0, 0.003, origin_directions.size()).to(args.device)
            origin_directions = F.normalize(origin_directions)
            scale = normalize_norm
            if args.remove_bad_clusters and bad_clusters:
                aggregation_weights = new_weights.clone()
                cluster_to_reserve_idx = {}
                for sim_idx in sim_idxs:
                    sim_idx = list(sim_idx)
                    sim_idx.sort()
                    if participant_indices is not None:
                        global_cluster = [participant_indices[idx] for idx in sim_idx]
                        global_cluster_set = frozenset(global_cluster)
                        representative_local_idx = sim_idx[0]
                        if representative_local_idx in reserve_idxs:
                            reserve_idx = reserve_idxs.index(representative_local_idx)
                            cluster_to_reserve_idx[global_cluster_set] = reserve_idx
                weights_modified = False
                for bad_cluster in bad_clusters:
                    bad_cluster_set = frozenset(bad_cluster)
                    if bad_cluster_set in cluster_to_reserve_idx:
                        reserve_idx = cluster_to_reserve_idx[bad_cluster_set]
                        if reserve_idx < aggregation_weights.size(1):
                            aggregation_weights[0, reserve_idx] = 0.0
                            weights_modified = True
                if weights_modified:
                    if aggregation_weights.sum() > 0:
                        aggregation_weights = aggregation_weights / aggregation_weights.sum()
                    else:
                        aggregation_weights = new_weights

                    if args.collaborative_security and suspect_scores:
                        singleton_clients = identify_singleton_clients(global_sim_idxs, participant_indices)
                        punished_count = 0
                        for i, local_idx in enumerate(reserve_idxs):
                            if i < aggregation_weights.size(1):
                                if participant_indices is not None:
                                    client_id = participant_indices[local_idx]
                                else:
                                    client_id = local_idx
                                if client_id not in singleton_clients:
                                    continue
                                suspect_score = suspect_scores.get(client_id, 0)
                                client_malicious_rounds = 0
                                if conspiracy_graph is not None:
                                    client_malicious_rounds = conspiracy_graph.client_malicious_rounds.get(client_id,0)
                                if suspect_score >= args.suspect_threshold and client_malicious_rounds >= args.min_conspiracy_rounds:
                                    penalty = args.suspect_penalty_factor
                                    original_weight = aggregation_weights[0, i].item()
                                    aggregation_weights[0, i] *= penalty
                                    punished_count += 1
                        if aggregation_weights.sum() > 0:
                            aggregation_weights = aggregation_weights / aggregation_weights.sum()
                        else:
                            aggregation_weights = new_weights
                    principal_direction = torch.matmul(aggregation_weights, origin_directions * scale)
                else:
                    principal_direction = torch.matmul(new_weights, origin_directions * scale)
            else:
                principal_direction = torch.matmul(new_weights, origin_directions * scale)
            global_update[name] = principal_direction
    average_credit_scores = None
    if accumulated_credit_scores is not None:
        layer_count = sum(1 for name in model_updates.keys() if 'num_batches_tracked' not in name)
        average_credit_scores = []
        for i, total_score in enumerate(accumulated_credit_scores):
            average_score = total_score / layer_count if layer_count > 0 else 0.0
            average_credit_scores.append(average_score)
    else:
        print("None")
    return global_update, average_credit_scores, client_cluster_sizes, weighted_variance_rank_head, weighted_variance_rank_tail, global_sim_idxs,avg_reputation_rank_head, avg_reputation_rank_tail,conspiracy_graph

def foolsgold(model_updates):
    keys = list(model_updates.keys())
    last_layer_updates = model_updates[keys[-2]]
    K = len(last_layer_updates)
    cs = smp.cosine_similarity(last_layer_updates.cpu().numpy()) - np.eye(K)
    maxcs = np.max(cs, axis=1)
    # === pardoning ===
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    alpha = np.max(cs, axis=1)
    wv = 1 - alpha
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # === Rescale so that max value is wv ===
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # === Logit function ===
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    # === calculate global update ===
    global_update = defaultdict()
    for name in keys:
        tmp = None
        for i, j in enumerate(range(len(wv))):
            if i == 0:
                tmp = model_updates[name][j] * wv[j]
            else:
                tmp += model_updates[name][j] * wv[j]
        global_update[name] = 1 / len(wv) * tmp

    return global_update


def flame(trained_params, current_model_param, param_updates):
    # === clustering ===
    trained_params = torch.stack(trained_params).double()
    cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic",
                              min_cluster_size=args.participant_sample_size // 2 + 1,
                              min_samples=1, allow_single_cluster=True)
    cluster.fit(trained_params)
    predict_good = []
    for i, j in enumerate(cluster.labels_):
        if j == 0:
            predict_good.append(i)
    k = len(predict_good)

    # === median clipping ===
    model_updates = trained_params[predict_good] - current_model_param
    local_norms = torch.norm(model_updates, dim=1)
    S_t = torch.median(local_norms)
    scale = S_t / local_norms
    scale = torch.where(scale > 1, torch.ones_like(scale), scale)
    model_updates = model_updates * scale.view(-1, 1)

    # === aggregating ===
    trained_params = current_model_param + model_updates
    trained_params = trained_params.sum(dim=0) / k

    # === noising ===
    delta = 1 / (args.participant_sample_size ** 2)
    epsilon = 10000
    lambda_ = 1 / epsilon * (math.sqrt(2 * math.log((1.25 / delta))))
    sigma = lambda_ * S_t.numpy()
    print(f"sigma: {sigma}; #clean models / clean models: {k} / {predict_good}, median norm: {S_t},")
    trained_params.add_(torch.normal(0, sigma, size=trained_params.size()))

    # === bn ===
    global_update = dict()
    for i, (name, param) in enumerate(param_updates.items()):
        if 'num_batches_tracked' in name:
            global_update[name] = 1 / k * \
                                  param_updates[name][predict_good].sum(dim=0, keepdim=True)
        elif 'running_mean' in name or 'running_var' in name:
            local_norms = torch.norm(param_updates[name][predict_good], dim=1)
            S_t = torch.median(local_norms)
            scale = S_t / local_norms
            scale = torch.where(scale > 1, torch.ones_like(scale), scale)
            global_update[name] = param_updates[name][predict_good] * scale.view(-1, 1)
            global_update[name] = 1 / k * global_update[name].sum(dim=0, keepdim=True)

    return trained_params.float().to(args.device), global_update


def fltrust(model_updates, param_updates, clean_param_update):
    cos = torch.nn.CosineSimilarity(dim=0)
    g0_norm = torch.norm(clean_param_update)
    weights = []
    for param_update in param_updates:
        weights.append(F.relu(cos(param_update.view(-1, 1), clean_param_update.view(-1, 1))))
    weights = torch.tensor(weights).to(args.device).view(1, -1)
    weights = weights / weights.sum()
    weights = torch.where(weights[0].isnan(), torch.zeros_like(weights), weights)
    nonzero_weights = torch.count_nonzero(weights.flatten())
    nonzero_indices = torch.nonzero(weights.flatten()).flatten()

    print(f'g0_norm: {g0_norm}, '
          f'weights_sum: {weights.sum()}, '
          f'*** {nonzero_weights} *** model updates are considered to be aggregated !')

    normalize_weights = []
    for param_update in param_updates:
        normalize_weights.append(g0_norm / torch.norm(param_update))

    global_update = dict()
    for name, params in model_updates.items():
        if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
            global_update[name] = 1 / nonzero_weights * params[nonzero_indices].sum(dim=0, keepdim=True)
        else:
            global_update[name] = torch.matmul(
                weights,
                params * torch.tensor(normalize_weights).to(args.device).view(-1, 1))
    return global_update


def robust_lr(model_updates):
    global_update = dict()
    for name, param in model_updates.items():
        if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
            global_update[name] = 1 / args.participant_sample_size * \
                                  model_updates[name].sum(dim=0, keepdim=True)
        else:
            signs = torch.sign(model_updates[name])
            sm_of_signs = torch.abs(torch.sum(signs, dim=0, keepdim=True))
            sm_of_signs[sm_of_signs < args.robustLR_threshold] = -1
            sm_of_signs[sm_of_signs >= args.robustLR_threshold] = 1
            global_update[name] = 1 / args.participant_sample_size * \
                                  (sm_of_signs * model_updates[name].sum(dim=0, keepdim=True))
    return global_update
