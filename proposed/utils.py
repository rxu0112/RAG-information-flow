import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from kneed import KneeLocator
from scipy.stats import wasserstein_distance
from scipy.stats import entropy

def unpack_shap_explanation_contents(shap_values):
    values = getattr(shap_values, "hierarchical_values", None)
    if values is None:
        values = shap_values.values
    clustering = getattr(shap_values, "clustering", None)

    return np.array(values), clustering

def process_shap_values(tokens, values, grouping_threshold, separator, clustering=None, return_meta_data=False):
    # See if we got hierarchical input data. If we did then we need to reprocess the
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(values) != M:
        # make sure we were given a partition tree
        if clustering is None:
            raise ValueError(
                "The length of the attribution values must match the number of "
                "tokens if shap_values.clustering is None! When passing hierarchical "
                "attributions the clustering is also required."
            )

        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(values))
        lower_values[:M] = values[:M]
        max_values = np.zeros(len(values))
        max_values[:M] = np.abs(values[:M])
        for i in range(clustering.shape[0]):
            li = int(clustering[i, 0])
            ri = int(clustering[i, 1])
            groups.append(groups[li] + groups[ri])
            lower_values[M + i] = lower_values[li] + lower_values[ri] + values[M + i]
            max_values[i + M] = max(abs(values[M + i]) / len(groups[M + i]), max_values[li], max_values[ri])

        # compute the upper_values
        upper_values = np.zeros(len(values))

        def lower_credit(upper_values, clustering, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = int(clustering[i - M, 0])
            ri = int(clustering[i - M, 1])
            upper_values[i] = value
            value += values[i]
            #             lower_credit(upper_values, clustering, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
            #             lower_credit(upper_values, clustering, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, clustering, li, value * 0.5)
            lower_credit(upper_values, clustering, ri, value * 0.5)

        lower_credit(upper_values, clustering, len(values) - 1)

        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_values = []
        group_sizes = []

        # meta data
        token_id_to_node_id_mapping = np.zeros((M,))
        collapsed_node_ids = []

        def merge_tokens(new_tokens, new_values, group_sizes, i):
            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)

                # meta data
                collapsed_node_ids.append(i)
                token_id_to_node_id_mapping[i] = i

            else:
                # compute the dividend at internal nodes
                li = int(clustering[i - M, 0])
                ri = int(clustering[i - M, 1])
                dv = abs(values[i]) / len(groups[i])

                # if the interaction level is too high then just treat this whole group as one token
                if max(max_values[li], max_values[ri]) < dv * grouping_threshold:
                    new_tokens.append(
                        separator.join([tokens[g] for g in groups[li]])
                        + separator
                        + separator.join([tokens[g] for g in groups[ri]])
                    )
                    new_values.append(group_values[i])
                    group_sizes.append(len(groups[i]))

                    # setting collapsed node ids and token id to current node id mapping metadata

                    collapsed_node_ids.append(i)
                    for g in groups[li]:
                        token_id_to_node_id_mapping[g] = i

                    for g in groups[ri]:
                        token_id_to_node_id_mapping[g] = i

                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)

        merge_tokens(new_tokens, new_values, group_sizes, len(group_values) - 1)

        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        values = np.array(new_values)
        group_sizes = np.array(group_sizes)

        # meta data
        token_id_to_node_id_mapping = np.array(token_id_to_node_id_mapping)
        collapsed_node_ids = np.array(collapsed_node_ids)

        M = len(tokens)
    else:
        group_sizes = np.ones(M)
        token_id_to_node_id_mapping = np.arange(M)
        collapsed_node_ids = np.arange(M)

    if return_meta_data:
        return tokens, values, group_sizes, token_id_to_node_id_mapping, collapsed_node_ids
    else:
        return tokens, values, group_sizes


def to_simplex_shift(x):
    x = np.asarray(x, dtype=float)
    y = x - x.min()                 # make >= 0
    s = y.sum()
    return y / s

def is_word_or_number(token):
    return any(ch.isalnum() for ch in token)

def merge_tokens_with_values_with_capitalized(tokens, values):
    """
    Merge tokens as in merge_tokens, and sum the corresponding values.
    Returns:
        merged_tokens: list of merged tokens
        summed_values: list of summed values, same length as merged_tokens
    """
    import string

    assert len(tokens) == len(values), "tokens and values must have the same length"

    # Step 1: Merge subwords by checking if token starts with a space
    merged_tokens = []
    summed_values = []
    buffer_token = ""
    buffer_value = 0
    for t, v in zip(tokens, values):
        if t.startswith(" "):
            if buffer_token:
                merged_tokens.append(buffer_token)
                summed_values.append(buffer_value)
            buffer_token = t
            buffer_value = v
        else:
            buffer_token += t
            buffer_value += v
    if buffer_token:
        merged_tokens.append(buffer_token)
        summed_values.append(buffer_value)

    # Step 2: Merge consecutive capitalized tokens (skip if punctuation in between)
    final_tokens = []
    final_values = []
    i = 0
    while i < len(merged_tokens):
        t = merged_tokens[i]
        v = summed_values[i]
        if t.strip() and t.strip()[0].isupper():
            # Start of potential name
            name_tokens = [t]
            name_values = [v]
            j = i + 1
            while (
                j < len(merged_tokens)
                and merged_tokens[j].strip()
                and merged_tokens[j].strip()[0].isupper()
                and not any(p in merged_tokens[j-1] for p in [",", ".", ";"])
            ):
                name_tokens.append(merged_tokens[j])
                name_values.append(summed_values[j])
                j += 1
            final_tokens.append("".join(name_tokens))
            final_values.append(sum(name_values))
            i = j
        else:
            final_tokens.append(t)
            final_values.append(v)
            i += 1

    return final_tokens, final_values

def merge_tokens_with_values(tokens, values):
    """
    Merge tokens as in merge_tokens, and sum the corresponding values.
    Returns:
        merged_tokens: list of merged tokens
        summed_values: list of summed values, same length as merged_tokens
    """
    import string

    assert len(tokens) == len(values), "tokens and values must have the same length"

    # Step 1: Merge subwords by checking if token starts with a space
    merged_tokens = []
    summed_values = []
    buffer_token = ""
    buffer_value = 0
    for t, v in zip(tokens, values):
        if t.startswith(" "):
            if buffer_token:
                merged_tokens.append(buffer_token)
                summed_values.append(buffer_value)
            buffer_token = t
            buffer_value = v
        else:
            buffer_token += t
            buffer_value += v
    if buffer_token:
        merged_tokens.append(buffer_token)
        summed_values.append(buffer_value)

    return merged_tokens, summed_values


def top_indices_groups(values, top=5, desired_groups=5):
    """
    values: 1D array of scores
    top: initial number of top elements to select
    desired_groups: number of groups to form
    """
    sorted_indices = np.argsort(values)[::-1]  # descending
    selected_indices = list(sorted_indices[:top])

    def form_groups(indices):
        """Group consecutive indices."""
        indices = sorted(indices)
        groups = []
        for idx in indices:
            if groups and idx == groups[-1][-1] + 1:
                groups[-1].append(idx)
            else:
                groups.append([idx])
        return groups

    groups = form_groups(selected_indices)
    next_idx_ptr = top  # pointer to next top index to pick

    while len(groups) < desired_groups and next_idx_ptr < len(values):
        # add next top index
        selected_indices.append(sorted_indices[next_idx_ptr])
        next_idx_ptr += 1
        # re-form groups with the new set
        groups = form_groups(selected_indices)

    return groups


def enumerate_paths(matrices):
    """
    Enumerate all paths that end at the last node in the last layer.

    Args:
        matrices: list of (n x n) numpy arrays, lower-triangular.

    Returns:
        paths: list of (path, contribution) where path is a tuple of node indices
               starting from layer 1 to the final layer.
    """
    num_layers = len(matrices) + 1
    n = matrices[0].shape[0]
    target_node = n - 1  # last node index (0-based)

    results = []

    def dfs(layer, node, path, contrib):
        if layer == num_layers - 2:
            M = matrices[layer]
            results.append((tuple(path + [target_node]), contrib * M[target_node, node]))
            return

        M = matrices[layer]
        for next_node in range(node, n):  # only a <= b
            weight = M[next_node, node]
            dfs(layer + 1, next_node, path + [next_node], contrib * weight)

    # start from all possible nodes in first layer
    for start in range(n):
        dfs(0, start, [start], 1.0)

    return results

def plot_information_flow_route(nodes, edges, seq_len, prompt_tokens):
    """
    Visualize information flow with only 'attn_in' and 'attn_res' nodes,
    and edges representing attention and MLP contributions.

    Args:
        nodes: set of (block, token, stage) tuples with stage in {'attn_in', 'attn_res'}
        edges: list of (from_node, to_node, kind) where kind ∈ {
            'attn_residual_connection', 'token_contribution', 'mlp_contribution', 'mlp_residual'
        }
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([(src, tgt) for src, tgt, _ in edges])

    # Layout: block on x-axis, token on y-axis with vertical offset for 'attn_in' and 'attn_res'
    stage_x_offset = {
        'attn_in': - 0.25,
        'attn_res': 0.25,
    }
    pos = {(b, t, stage): (b+stage_x_offset[stage], -t) for (b, t, stage) in nodes}

    fig, ax = plt.subplots(figsize=(16, 9))

    # Node colors
    color_map = {
        'attn_in': 'skyblue',
        'attn_res': 'lightgreen',
    }
    for stage in ['attn_in', 'attn_res']:
        stage_nodes = [n for n in nodes if n[2] == stage]
        nx.draw_networkx_nodes(
            G, pos, nodelist=stage_nodes,
            node_color=color_map[stage], label=stage,
            node_size=100, alpha=0.9, ax=ax
        )

    # Define edge styles and curvature
    edge_styles = {
        'token_contribution': ('blue', 'dashed', 0.0),
        'attn_residual_connection': ('green', 'solid', -0.3),
        'mlp_contribution': ('red', 'dashed', 0.0),
        'mlp_residual_connection': ('orange', 'solid', -0.3),
    }

    # Draw edges
    for src, tgt, kind in edges:
        color, style, rad = edge_styles.get(kind, ('gray', 'solid', 0.0))
        ax.annotate(
            "",
            xy=pos[tgt],
            xytext=pos[src],
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                linestyle=style,
                linewidth=1.5,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )

    # Layer labels
    all_blocks = sorted(set(b for b, _, _ in nodes))
    for b in all_blocks[0:-1]:
        ax.text(b+ 0.2, -seq_len -1, f"L{b}", ha="center", va="center",
                fontsize=10, bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.6))
    # Token labels
    for i, tok in enumerate(prompt_tokens):
        ax.text(-1, -i, tok, ha="right", va="center", fontsize=5, fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.5))

    ax.set_ylim(-seq_len,1)
    plt.show()

class OrderedSet:
    def __init__(self):
        self._data = {}

    def add(self, item):
        self._data[item] = None  # overwrite if exists (no duplicate)

    def __iter__(self):
        return iter(self._data.keys())

    def __contains__(self, item):
        return item in self._data

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"OrderedSet({list(self._data.keys())})"

def find_next_node(candidates, history):
    """
    Find the global maximum from dict of candidate arrays,
    skipping visited (group, arr_idx, pos_idx).

    Args:
        candidates: dict[str, list[np.ndarray]]
            key = group name
            value = list of arrays
        history: set of (group, arr_idx, pos_idx) to skip

    Returns:
        (group, arr_idx, pos_idx, value) or (None, None, None, -inf)
    """
    max_val = -np.inf
    max_group = None
    max_arr = None
    max_pos = None

    for group, arrays in candidates.items():
        for j, arr in enumerate(arrays):  # loop over arrays in this group
            for k, val in enumerate(arr):  # loop over elements
                if (group, j, k) in history:
                    continue
                if val > max_val:
                    max_val = val
                    max_group, max_arr, max_pos = group, j, k

    return max_group, max_arr, max_pos, max_val

def build_heap(candidates):
    heap = []
    for group, arrays in candidates.items():
        for j, arr in enumerate(arrays):
            for k, val in enumerate(arr):
                heapq.heappush(heap, (-val, group, j, k))  # negative for max-heap
    return heap

def find_next_node_heap(heap, history):
    while heap:
        neg_val, group, j, k = heapq.heappop(heap)
        if (group, j, k) in history:
            continue
        return group, j, k, -neg_val
    return None, None, None, -np.inf

def outlier_topness(arr, arr2, agg="mean"):
    # Step 1: detect outliers in arr
    sorted_arr = np.sort(arr)[::-1]
    x = np.arange(len(sorted_arr))
    knee = KneeLocator(x, sorted_arr, curve="convex", direction="decreasing").knee
    if knee is None:  # fallback if knee not found
        return None

    threshold_val = sorted_arr[knee]
    outlier_indices = np.where(arr >= threshold_val)[0]

    # Step 2: rank arr2
    ranks = arr2.argsort()[::-1].argsort() + 1  # 1 = highest rank
    normalized_topness = 1 - (ranks - 1) / (len(arr2) - 1)

    # Step 3: collect topness values
    scores = normalized_topness[outlier_indices]

    # Step 4: aggregate into a single metric
    if agg == "mean":
        return scores.mean()
    elif agg == "median":
        return np.median(scores)
    elif agg == "max":
        return scores.max()
    else:
        raise ValueError("agg must be one of {'mean', 'median', 'sum'}")

def weighted_outlier_topness(arr, arr2):
    # Step 1: detect outliers in arr
    sorted_arr = np.sort(arr)[::-1]
    x = np.arange(len(sorted_arr))
    knee = KneeLocator(x, sorted_arr, curve="convex", direction="decreasing").knee
    if knee is None:
        return None

    threshold_val = sorted_arr[knee]
    outlier_indices = np.where(arr >= threshold_val)[0]

    # Step 2: compute topness in arr2
    ranks = arr2.argsort()[::-1].argsort() + 1  # 1 = best rank
    topness = 1 - (ranks - 1) / (len(arr2) - 1)

    # Step 3: assign weights from arr (normalize for stability)
    weights = arr[outlier_indices]
    weights = weights / weights.sum()

    # Step 4: weighted topness
    return np.dot(weights, topness[outlier_indices])


def knee_recall(arr, arr2):
    sorted_arr = np.sort(arr)[::-1]
    x = np.arange(len(sorted_arr))
    knee = KneeLocator(x, sorted_arr, curve="convex", direction="decreasing").knee
    if knee is None:
        return None

    threshold_val = sorted_arr[knee]
    outlier_indices = np.where(arr >= threshold_val)[0]

    # get top-knee items in arr2
    k = len(outlier_indices)
    topk_in_arr2 = arr2.argsort()[::-1][:k]

    # recall@knee
    overlap = len(set(outlier_indices) & set(topk_in_arr2))
    return overlap / len(outlier_indices)

# def entropy(probs, base=2):
#     probs = np.array(probs, dtype=float)
#     # Avoid log(0) by masking
#     probs = probs[probs > 0]
#     return -np.sum(probs * np.log(probs) / np.log(base))
#
# def positional_entropy(probs):
#     probs = np.array(probs, dtype=float)
#     probs = probs / probs.sum()  # normalize to sum=1
#
#     n = len(probs)
#     # Distance matrix: |i - j|
#     idx = np.arange(n)
#     dist_matrix = np.abs(idx[:, None] - idx[None, :])
#
#     # Rao’s quadratic entropy
#     H = np.sum(probs[:, None] * probs[None, :] * dist_matrix)
#
#     # Normalize to [0,1] if desired
#     H = H / dist_matrix.max() if dist_matrix.max() > 0 else 0
#     return H

def W_distance_with_uniform(probs, normalization=False):
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    positions = np.arange(len(probs))

    # Raw Wasserstein distance
    w_dist = wasserstein_distance(
        positions, positions,
        probs, np.ones_like(probs) / len(probs)
    )
    if normalization:
        w_dist = w_dist / (len(probs) - 1)

    return w_dist

def W_distance_with_shap(p, q, normalization=False):
    p = np.asarray(p, float); p /= p.sum()
    q = np.asarray(q, float); q /= q.sum()
    x = np.arange(len(p))
    dist = wasserstein_distance(x, x, p, q)
    if normalization:
        dist /= (len(p)-1)
    return dist

def topK_overlap_pos(vec1, vec2, k=3):
    idx1 = sorted(range(len(vec1)), key=lambda i: vec1[i], reverse=True)[:k]
    idx2 = sorted(range(len(vec2)), key=lambda i: vec2[i], reverse=True)[:k]
    overlap = len(list(set(idx1) & set(idx2)))
    return overlap

def topK_overlap_token(vec1, tokens1, vec2, tokens2, k=3):
    idx1 = sorted(range(len(vec1)), key=lambda i: vec1[i], reverse=True)[:k]
    idx2 = sorted(range(len(vec2)), key=lambda i: vec2[i], reverse=True)[:k]
    overlap = len(list(set(idx1) & set(idx2)))
    for i in range(len(idx1)):
        if idx1[i] != idx2[i]:
            if tokens1[idx1[i]] == tokens2[idx2[i]]:
                overlap += 1
                print('match one more')
    return overlap

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    numerator = np.dot(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    return numerator / denominator


def KL_divergence_with_uniform(probs, base=np.e, normalization=False, eps=1e-12):
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()  # normalize

    n = len(probs)
    uniform = np.ones(n) / n

    probs = np.clip(probs, eps, 1)
    uniform = np.clip(uniform, eps, 1)

    probs /= probs.sum()  # renormalize after clipping
    uniform /= uniform.sum()


    kl = entropy(probs, qk=uniform, base=base)  # scipy’s entropy(p, q) = KL(p||q)

    if normalization:
        kl = kl / np.log(n)  # maximum KL occurs at a delta distribution

    return kl

def KL_divergence_with_shap(p, q, normalization=False, eps=1e-12):
    p = np.asarray(p, float)
    p /= p.sum()
    q = np.asarray(q, float)
    q /= q.sum()

    # Add epsilon to avoid log(0) or division by zero
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    p /= p.sum()  # renormalize after clipping
    q /= q.sum()

    kl = entropy(p, qk=q, base=np.e)

    if normalization:
        kl /= np.log(len(p))

    return kl

def find_section(tokens, start, keyword):
    matches = [i + start for i, t in enumerate(tokens[start:]) if t.strip() == keyword]
    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        return None
    return matches[0]

def compute_ece(scores, y_true, n_bins=10):
    """
    Expected Calibration Error (ECE).

    scores : array-like, predicted probabilities (between 0 and 1).
    y_true : array-like, true binary labels (0 or 1).
    n_bins : number of bins (default 10).
    """
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    total_count = len(scores)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # include right edge in the last bin
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)

        if not np.any(mask):
            continue

        avg_conf = scores[mask].mean()
        acc = y_true[mask].mean()
        bin_frac = mask.mean()

        # contribution to ECE
        ece += np.abs(acc - avg_conf) * bin_frac

    return ece

def coverage_at_accuracy(y_true, y_proba, target_acc=0.8):
    # Sort by confidence descending
    sorted_idx = np.argsort(-y_proba)
    y_true_sorted = y_true[sorted_idx]
    y_proba_sorted = y_proba[sorted_idx]

    # Cumulative accuracy
    correct_cumsum = np.cumsum(y_true_sorted)
    counts = np.arange(1, len(y_true_sorted) + 1)
    acc_cumsum = correct_cumsum / counts

    # Find max coverage where cumulative accuracy >= target_acc
    valid_idx = np.where(acc_cumsum >= target_acc)[0]
    if len(valid_idx) == 0:
        return 0.0  # cannot achieve target accuracy
    max_cov = (valid_idx[-1] + 1) / len(y_true)
    return max_cov