import numpy as np

def calculate_perplexity(probabilities):

    log_likelihood = np.sum(np.log(probabilities))   # 所有 log 概率的和
    avg_log_likelihood = -log_likelihood / len(probabilities)  # 平均负 log
    perplexity = np.exp(avg_log_likelihood)          # 指数化
    return perplexity


