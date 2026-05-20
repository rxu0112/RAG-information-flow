import json
import torch
from utils import *
from rbo import rbo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import optuna
from scipy.stats import spearmanr, pearsonr
import argparse

parser = argparse.ArgumentParser(description="train a calibrator using info-flow features")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
parser.add_argument('--reranker', type=str, choices=['Qwen3' 'MiniLM'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct', 'Llama-3-8B-Instruct', 'gemma-3-4b-it'], required=True)
parser.add_argument('--p', type=float, required=True)

args = parser.parse_args()


test_summary = torch.load(f"results/{args.model}_{args.dataset}/proposed_metric/{args.dataset}_test_uq_{args.reranker}_p={str(args.p)}.pt")
train_summary = torch.load(f"results/{args.model}_{args.dataset}/proposed_metric/{args.dataset}_train_uq_{args.reranker}_p={str(args.p)}.pt")
val_summary = torch.load(f"results/{args.model}_{args.dataset}/proposed_metric/{args.dataset}_val_uq_{args.reranker}_p={str(args.p)}.pt")

X_train, y_train = train_summary["X"], train_summary["y"]
X_test, y_test = test_summary["X"], test_summary["y"]
X_val, y_val = val_summary["X"], val_summary["y"]

hem_test = test_summary["hem"]

# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

scale_pos_weight = (len(y_train)-np.sum(y_train)) / np.sum(y_train)

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 1, 6),
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "lambda": trial.suggest_float("lambda", 0.1, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "seed": 42
    }

    evals = [(dtrain, "train"), (dval, "val")]

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    best_round = bst.best_iteration  # <-- best round for this trial
    y_proba_val = bst.predict(dval)
    val_auc = roc_auc_score(y_val, y_proba_val)
    val_auprc = average_precision_score(y_val, y_proba_val)
    score = val_auc+val_auprc

    # You can store best_round in trial user attributes
    trial.set_user_attr("best_round", best_round)

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best validation objective:", study.best_value)
print("Best params:", study.best_params)

dtrain_full = xgb.DMatrix(np.vstack([X_train, X_val]), label=np.hstack([y_train, y_val]))
bst = xgb.train(
    params=study.best_params,
    dtrain=dtrain_full,
    num_boost_round=study.best_trial.user_attrs["best_round"],
    verbose_eval=False
)
y_proba_test = bst.predict(dtest)

auprc = average_precision_score(y_test, y_proba_test)
print('test auprc:', auprc)
auroc = roc_auc_score(y_test, y_proba_test)
print('test auroc:', auroc)
y_pred = (y_proba_test >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print("Accuracy at threshold 0.5:", acc)
cov = coverage_at_accuracy(y_test, y_proba_test, target_acc=0.8)
print("Coverage at 80% accuracy:", cov)
corr, pval = spearmanr(hem_test, y_proba_test)
print("Spearman correlation:", corr)
print("p-value:", pval)
corr, pval = pearsonr(hem_test, y_proba_test)
print("Pearson correlation:", corr)
print("p-value:", pval)
ece = compute_ece(y_proba_test, y_test, n_bins=50)
print("ECE:", ece)

y_proba_train = bst.predict(dtrain)
auprc = average_precision_score(y_train, y_proba_train)
print('train auprc:', auprc)
auroc = roc_auc_score(y_train, y_proba_train)
print('train auroc:', auroc)

output = {'y_test': y_test.tolist(), 'y_proba_test': y_proba_test.tolist(), 'hem_test': hem_test.tolist()}
with open(f'results/{args.model}_{args.dataset}/proposed_{args.reranker}_p={str(args.p)}.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)


# --- only proposed metric vs only relevance score from reranker ---

dtrain_full = xgb.DMatrix(np.vstack([X_train[:, :-1], X_val[:, :-1]]),
                          label=np.hstack([y_train, y_val]))
dtest_subset = xgb.DMatrix(X_test[:, :-1])

bst = xgb.train(
    params=study.best_params,
    dtrain=dtrain_full,
    num_boost_round=study.best_trial.user_attrs["best_round"],
    verbose_eval=False
)
y_proba = bst.predict(dtest_subset)
auroc = roc_auc_score(y_test, y_proba)
print('test auroc only proposed metric:', auroc)

dtrain_full = xgb.DMatrix(np.hstack([X_train[:, -1], X_val[:, -1]]).reshape(-1,1),
                          label=np.hstack([y_train, y_val]))
dtest_subset = xgb.DMatrix(X_test[:, -1].reshape(-1,1))

bst = xgb.train(
    params=study.best_params,
    dtrain=dtrain_full,
    num_boost_round=study.best_trial.user_attrs["best_round"],
    verbose_eval=False
)
y_proba = bst.predict(dtest_subset)
auroc = roc_auc_score(y_test, y_proba)
print('test auroc only relevence score from reranker:', auroc)



# Permutation Importance

from copy import deepcopy
import pandas as pd

def permutation_importance_auc(model, X, y, metric_fn, n_repeats=10, seed=42):
    rng = np.random.default_rng(seed)

    X = X.copy()
    base_pred = model.predict(xgb.DMatrix(X))
    base_score = metric_fn(y, base_pred)

    importances = []

    for j in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()

            col = X_perm[:, j].copy()
            rng.shuffle(col)
            X_perm[:, j] = col

            pred = model.predict(xgb.DMatrix(X_perm))
            score = metric_fn(y, pred)
            scores.append(score)

        importance = base_score - np.mean(scores)
        importances.append(importance)

    return np.array(importances)

imp_auroc = permutation_importance_auc(bst, X_test, y_test, roc_auc_score)
imp_auprc = permutation_importance_auc(bst, X_test, y_test, average_precision_score)

