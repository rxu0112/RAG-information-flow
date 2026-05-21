import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
import json
import math
import torch
import xgboost as xgb
import optuna
import argparse

parser = argparse.ArgumentParser(description="a calibrated version of baselines")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct','Llama-3-8B-Instruct','gemma-3-4b-it'], required=True)

args = parser.parse_args()


print(args.model+" "+args.dataset)
with open(f'results/{args.model}_{args.dataset}/baselines/baselines_test.json', 'r', encoding='utf-8') as f:
    baselines_test = json.load(f)

with open(f'results/{args.model}_{args.dataset}/baselines/baselines_train.json', 'r', encoding='utf-8') as f:
    baselines_train = json.load(f)

with open(f'results/{args.model}_{args.dataset}/baselines/baselines_val.json', 'r', encoding='utf-8') as f:
    baselines_val = json.load(f)

test_summary = torch.load(f"results/{args.model}_{args.dataset}/proposed_metric/{args.dataset}_test_uq_Qwen3_p=0.7.pt")
train_summary = torch.load(f"results/{args.model}_{args.dataset}/proposed_metric/{args.dataset}_train_uq_Qwen3_p=0.7.pt")
val_summary = torch.load(f"results/{args.model}_{args.dataset}/proposed_metric/{args.dataset}_val_uq_Qwen3_p=0.7.pt")

optuna.logging.disable_default_handler()
for score in ['ppl_score','p_true_score','RE_score','SE_score','att_score', 'focus_score']: #'ppl_score','p_true_score','RE_score','SE_score','attention', 'focus'
    if score == 'focus_score':
        valid_train_mask = np.array(baselines_train[score]) != None
        valid_val_mask = np.array(baselines_val[score]) != None
        valid_test_mask = np.array(baselines_test[score]) != None
        train_valid = -np.array(baselines_train[score])[valid_train_mask].astype(float)
        X_train = np.array(train_valid).reshape(-1, 1)
        y_train = train_summary["y"][valid_train_mask]
        val_valid = -np.array(baselines_val[score])[valid_val_mask].astype(float)
        X_val = np.array(val_valid).reshape(-1, 1)
        y_val = val_summary["y"][valid_val_mask]
        test_valid = -np.array(baselines_test[score])[valid_test_mask].astype(float)
        X_test = np.array(test_valid).reshape(-1, 1)
        y_test = test_summary["y"][valid_test_mask]
    elif score == 'att_score':
        valid_train_mask = np.array(baselines_train[score]) != None
        valid_val_mask = np.array(baselines_val[score]) != None
        valid_test_mask = np.array(baselines_test[score]) != None
        train_valid = np.array(baselines_train[score])[valid_train_mask].astype(float)
        X_train = np.array(train_valid).reshape(-1, 1)
        y_train = train_summary["y"][valid_train_mask]
        val_valid = np.array(baselines_val[score])[valid_val_mask].astype(float)
        X_val = np.array(val_valid).reshape(-1, 1)
        y_val = val_summary["y"][valid_val_mask]
        test_valid = np.array(baselines_test[score])[valid_test_mask].astype(float)
        X_test = np.array(test_valid).reshape(-1, 1)
        y_test = test_summary["y"][valid_test_mask]
    else:
        X_train = np.array(baselines_train[score]).reshape(-1,1)
        X_val = np.array(baselines_val[score]).reshape(-1, 1)
        X_test = np.array(baselines_test[score]).reshape(-1, 1)
        y_train = train_summary["y"]
        y_test = test_summary["y"]
        y_val = val_summary["y"]

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
    study.optimize(objective, n_trials=10)

    # print("Best validation objective:", study.best_value)
    # print("Best params:", study.best_params)
    dtrain_full = xgb.DMatrix(np.vstack([X_train,X_val]),
                              label=np.hstack([y_train, y_val]))

    bst = xgb.train(
        params=study.best_params,
        dtrain=dtrain,
        num_boost_round=study.best_trial.user_attrs["best_round"],
        verbose_eval=False
    )
    y_proba_test = bst.predict(dtest)

    print(score)
    auroc = roc_auc_score(y_test, X_test)
    auprc = average_precision_score(y_test, X_test)
    print('original test auroc:', auroc,', original test auprc:', auprc)
    auroc = roc_auc_score(y_test, y_proba_test)
    auprc = average_precision_score(y_test, y_proba_test)
    print('calibrated test auroc:', auroc, ', calibrated test auprc:', auprc)
