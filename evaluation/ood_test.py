import json
import torch
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
import numpy as np

parser = argparse.ArgumentParser(description="OOD test")
parser.add_argument('--dataset_train', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct','Llama-3-8B-Instruct','gemma-3-4b-it'], required=True)

args = parser.parse_args()

train_summary = torch.load(f"results/{args.model}_{args.dataset_train}/proposed_metric/{args.dataset_train}_train_uq_Qwen3_p=0.7.pt")
val_summary = torch.load(f"results/{args.model}_{args.dataset_train}/proposed_metric/{args.dataset_train}_val_uq_Qwen3_p=0.7.pt")

X_train, y_train = train_summary["X"], train_summary["y"]
X_val, y_val = val_summary["X"], val_summary["y"]


# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


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

print(args.model)
print("training data: "+args.dataset_train)
datasets_test = ['squad2','hotpot','msmarco']
for dataset_test in datasets_test:
    test_summary = torch.load(f"results/{args.model}_{dataset_test}/proposed_metric/{dataset_test}_test_uq_Qwen3_p=0.7.pt")
    X_test, y_test = test_summary["X"], test_summary["y"]
    hem_test = test_summary["hem"]
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_proba_test = bst.predict(dtest)
    auprc = average_precision_score(y_test, y_proba_test)
    print(dataset_test+' test auprc:', auprc)
    auroc = roc_auc_score(y_test, y_proba_test)
    print(dataset_test+' test auroc:', auroc)