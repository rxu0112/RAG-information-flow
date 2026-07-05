import numpy as np
import scipy.stats as stats
from collections import OrderedDict
from .rank_metrics import ndcg_at_k
from torch import nn
import torch
from ragu.passage_utility.reward_learner.bayesian_bert import BayesainBert

def evaluateReward(learnt_values, ref_values, short=False, top_answer=None):
    metrics_dic = OrderedDict()
    # if not short:
    #     ### compute the absolute errors
    #     # mse = mean_squared_error(ref_values,learnt_values)
    #     # metrics_dic['mse'] = mse

    #     ### compute KL divergence
    #     #js = jsd(learnt_values,ref_values)
    #     #metrics_dic['jsd-original'] = js
    #     prob_optimal = getSoftmaxList(ref_values, 1.0)
    #     prob_learnt = getSoftmaxList(learnt_values, 1.0)
    #     js = jsd(prob_optimal,prob_learnt)
    #     metrics_dic['jsd-softmax'] = js
        #kld = stats.entropy(prob_optimal, prob_learnt)
        #metrics_dic['kld'] = kld

    ### compute Kendall's tau, Spearman's rho and Pearson correlation coefficient
    tau, _ = stats.kendalltau(learnt_values, ref_values)
    rho, _ = stats.spearmanr(learnt_values, ref_values)
    pcc, _ = stats.pearsonr(learnt_values, ref_values)
    metrics_dic['tau'] = tau
    metrics_dic['rho'] = rho
    metrics_dic['pcc'] = pcc

    ### compute nDCG
    ll = np.array(ref_values)[np.flip(np.argsort(learnt_values), 0)]

    ndcg = ndcg_at_k(ll,int(0.01*len(ll)))
    metrics_dic['ndcg_at_1%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.05*len(ll)))
    metrics_dic['ndcg_at_5%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.1*len(ll)))
    metrics_dic['ndcg_at_10%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.2*len(ll)))
    metrics_dic['ndcg_at_20%'] = ndcg
    ndcg = ndcg_at_k(ll,int(0.5*len(ll)))
    metrics_dic['ndcg_at_50%'] = ndcg
    ndcg = ndcg_at_k(ll,len(ll))
    metrics_dic['ndcg_at_all'] = ndcg

    metrics_dic['score_of_estimated_best'] = ref_values[np.argmax(learnt_values)]
    metrics_dic['score_of_true_best'] = np.max(ref_values)

    ranked_items = np.argsort(learnt_values) # smallest score first
    metrics_dic['rank_of_best'] = float(len(learnt_values) - np.argwhere(ranked_items == np.argmax(ref_values)).flatten()[0])

    if top_answer is not None:
        #accuracy in matching the top items -- when averaged across topics, it will become an accuracy score
        metrics_dic['accuracy'] = 100 * float(top_answer == np.argmax(learnt_values))

    return metrics_dic





if __name__ == '__main__' :
    pass
    # model_path = ''
    # checkpoint = torch.load(model_path)
    # bb = BayesainBert(base_model=base_model, subspace='covariance', max_num_models=args.max_num_models,
    #                   device=device, lr_init=args.lr_init, momentum=args.momentum, weight_decay=args.wd,
    #                   swag_start=args.swag_start, swag_lr=args.swag_lr, swag_c_epochs=args.swag_c_epochs,
    #                   epochs=args.epochs, batch_size=args.batch_size, pretrained_model=args.pretrained_model)
    # model.load_state_dict(checkpoint['state_dict'])