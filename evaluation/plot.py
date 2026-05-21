import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
import json
from utils import *
import math
import argparse

parser = argparse.ArgumentParser(description="present main results")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct', 'Llama-3-8B-Instruct', 'gemma-3-4b-it'], required=True)
parser.add_argument('--reranker', type=str, choices=['Qwen3' 'MiniLM','BGE'], required=True)
parser.add_argument('--p', type=float, required=True)

args = parser.parse_args()

with open(f'results/{args.model}_{args.dataset}/baselines/baselines_test.json', 'r', encoding='utf-8') as f:
    baselines = json.load(f)

with open(f'results/{args.model}_{args.dataset}/proposed_{args.reranker}_p={str(args.p)}.json', 'r', encoding='utf-8') as f:
    proposed = json.load(f)


se_auprc = average_precision_score(baselines['y_true'], baselines['SE_score'])
se_auroc = roc_auc_score(baselines['y_true'], baselines['SE_score'])
se_pearson, _ = pearsonr(baselines['hem_score'], baselines['SE_score'])
se_spearman, _ = spearmanr(baselines['hem_score'], baselines['SE_score'])

re_auprc = average_precision_score(baselines['y_true'], baselines['RE_score'])
re_auroc = roc_auc_score(baselines['y_true'], baselines['RE_score'])
re_pearson, _ = pearsonr(baselines['hem_score'], baselines['RE_score'])
re_spearman, _ = spearmanr(baselines['hem_score'], baselines['RE_score'])

p_true_auprc = average_precision_score(baselines['y_true'], baselines['p_true_score'])
p_true_auroc = roc_auc_score(baselines['y_true'], baselines['p_true_score'])
p_true_pearson, _ = pearsonr(baselines['hem_score'], baselines['p_true_score'])
p_true_spearman, _ = spearmanr(baselines['hem_score'], baselines['p_true_score'])
p_true_cali_acc = accuracy_score(baselines['y_true'], [1 if s > 0.5 else 0 for s in baselines['p_true_score']])
p_true_cov_acc = coverage_at_accuracy(np.array(baselines['y_true']), np.array(baselines['p_true_score']), target_acc=0.8)

ppl_auprc = average_precision_score(baselines['y_true'], baselines['ppl_score'])
ppl_auroc = roc_auc_score(baselines['y_true'], baselines['ppl_score'])
ppl_pearson, _ = pearsonr(baselines['hem_score'], baselines['ppl_score'])
ppl_spearman, _ = spearmanr(baselines['hem_score'], baselines['ppl_score'])

valid_att_mask = np.array(baselines['att_score']) != None
att_test_valid = np.array(baselines['att_score'])[valid_att_mask].astype(float)
att_auprc = average_precision_score(np.array(baselines['y_true'])[valid_att_mask], att_test_valid)
att_auroc = roc_auc_score(np.array(baselines['y_true'])[valid_att_mask], att_test_valid)
att_pearson, _ = pearsonr(np.array(baselines['hem_score'])[valid_att_mask], att_test_valid)
att_spearman, _ = spearmanr(np.array(baselines['hem_score'])[valid_att_mask], att_test_valid)

valid_focus_mask = np.array(baselines['focus_score']) != None
focus_test_valid = -np.array(baselines['focus_score'])[valid_focus_mask].astype(float)
focus_auprc = average_precision_score(np.array(baselines['y_true'])[valid_focus_mask], focus_test_valid)
focus_auroc = roc_auc_score(np.array(baselines['y_true'])[valid_focus_mask], focus_test_valid)
focus_pearson, _ = pearsonr(np.array(baselines['hem_score'])[valid_focus_mask], focus_test_valid)
focus_spearman, _ = spearmanr(np.array(baselines['hem_score'])[valid_focus_mask], focus_test_valid)

utility_auprc = average_precision_score(baselines['y_true'], baselines['utility_score'])
utility_auroc = roc_auc_score(baselines['y_true'], baselines['utility_score'])
utility_pearson, _ = pearsonr(baselines['hem_score'], baselines['utility_score'])
utility_spearman, _ = spearmanr(baselines['hem_score'], baselines['utility_score'])

embed_auprc = average_precision_score(baselines['y_true'], baselines['embed_score'])
embed_auroc = roc_auc_score(baselines['y_true'], baselines['embed_score'])
embed_pearson, _ = pearsonr(baselines['hem_score'], baselines['embed_score'])
embed_spearman, _ = spearmanr(baselines['hem_score'], baselines['embed_score'])
embed_cali_acc = accuracy_score(baselines['y_true'], [1 if s > 0.5 else 0 for s in baselines['embed_score']])
embed_cov_acc = coverage_at_accuracy(np.array(baselines['y_true']), np.array(baselines['embed_score']), target_acc=0.8)

proposed_auprc = average_precision_score(proposed['y_test'], proposed['y_proba_test'])
proposed_auroc = roc_auc_score(proposed['y_test'], proposed['y_proba_test'])
proposed_pearson, _ = pearsonr(proposed['hem_test'], proposed['y_proba_test'])
proposed_spearman, _ = spearmanr(proposed['hem_test'], proposed['y_proba_test'])
proposed_cali_acc = accuracy_score(proposed['y_test'], [1 if s > 0.5 else 0 for s in proposed['y_proba_test']])
proposed_cov_acc = coverage_at_accuracy(np.array(proposed['y_test']), np.array(proposed['y_proba_test']), target_acc=0.8)

methods_scores = {
    'PPL': baselines['ppl_score'],
    'P(True)': baselines['p_true_score'],
    'Regular Entropy': baselines['RE_score'],
    'Semantic Entropy': baselines['SE_score'],
    'KnowingMore': baselines['embed_score'],
    'Utility Ranker': baselines['utility_score'],
    'Attention': baselines['att_score'],
    'Focus': -np.array(baselines['focus_score'])[valid_focus_mask].astype(float),
    'Ours': proposed['y_proba_test'],
}

colors = {
    'Regular Entropy': '#1f77b4',     # dark blue
    'Semantic Entropy': '#aec7e8',    # light blue
    'PPL': '#ff7f0e',                 # dark orange
    'P(True)': '#ffbb78',             # light orange
    'KnowingMore': '#d62728',         # dark red
    'Utility Ranker': '#ff9896',      # light red
    'Attention': '#9467bd',           # purple
    'Focus': '#8c564b',               # brown
    'Ours': '#1abc9c'                 # teal
}

y_true_baselines = baselines['y_true']
y_true_proposed = proposed['y_test']

# --- ROC Curve ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for method, scores in methods_scores.items():
    if method == 'Ours':
        y_true = y_true_proposed
    elif method == 'Focus':
        y_true = np.array(y_true_baselines)[valid_focus_mask]
    elif method == 'Attention':
        y_true = np.array(y_true_baselines)[valid_att_mask]
    else:
        y_true = y_true_baselines
        
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.plot(fpr, tpr, label=f'{method}', color=colors[method])

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate',  fontsize=14)
plt.ylabel('True Positive Rate',  fontsize=14)
plt.title('ROC Curves')
plt.legend()
plt.grid(True)

# --- Precision-Recall Curve ---
plt.subplot(1, 3, 2)
for method, scores in methods_scores.items():
    if method == 'Ours':
        y_true = y_true_proposed
    elif method == 'Focus':
        y_true = np.array(y_true_baselines)[valid_focus_mask]
    elif method == 'Attention':
        y_true = np.array(y_true_baselines)[valid_att_mask]
    else:
        y_true = y_true_baselines

    precision, recall, _ = precision_recall_curve(y_true, scores)
    plt.plot(recall, precision, label=f'{method}',color=colors[method])

plt.xlabel('Recall',  fontsize=14)
plt.ylabel('Precision',  fontsize=14)
plt.title('PR Curves')
plt.legend(ncol=2)
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Accuracy vs Confidence (Calibration Plot) ---
plt.subplot(1, 3, 3)

for method in ['P(True)', 'KnowingMore', 'Ours']:
    scores = methods_scores[method]
    y_true = y_true_proposed if method == 'Ours' else y_true_baselines

    # Bin predictions by confidence
    bins = np.linspace(0, 1, 11)  # 10 bins
    bin_indices = np.digitize(scores, bins) - 1

    bin_acc = []
    bin_conf = []

    for b in range(len(bins) - 1):
        mask = bin_indices == b
        if np.any(mask):
            acc = np.mean(np.array(y_true)[mask])
            bin_center = (bins[b] + bins[b + 1]) / 2
            bin_acc.append(acc)
            bin_conf.append(bin_center)

    plt.plot(bin_conf, bin_acc, marker='o', label=method, color=colors[method])

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
plt.xlabel('Confidence', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Accuracy vs Confidence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

for method in ['P(True)','KnowingMore', 'Ours']:
    scores = np.asarray(methods_scores[method])
    y_true = np.asarray(y_true_proposed) if method == 'Ours' else np.asarray(y_true_baselines)

    ece = compute_ece(scores, y_true, n_bins=10)
    print(f"{method}: ECE = {ece:.4f}")