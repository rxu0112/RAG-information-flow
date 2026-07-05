import os
import glob
import numpy as np
import re
from Other_Baselines.ragu.passage_utility.utils.misc import sigmoid
import sys
sys.path.append('../utils')
from Other_Baselines.ragu.utils.utils import load_jsonlines, load_file, save_file_jsonl
import random

def getPref(score1, score2, temperature):
    """To simulate user choice, TODO: not used for the moment, see if necessary for active learning. """
    # print('idx1 = %i, idx2 = %i, available reference vals =%i' % (idx1, idx2, len(self.ref_values)))
    # prob = sigmoid(self.ref_values[question_id][idx1]-self.ref_values[question_id][idx2], self.temperature)
    prob = sigmoid(score1-score2, temperature)

    if random.random() <= prob:
        return 1  # summary1 is preferred
    else:
        return -1  # summary2 is preferred

def construct_dataset(data, top_n, single_net, criteria='retriever'):
    '''Used for interactive learning and test mode. Here a silver reference ranking 
       is computed in the same way as the score used for training. These could be replaced by different, e.g.,
       human ranking criteria, the same LLM judge or other LLM.
       We can precompute this on the test set, we assing a reference ranking to passages.
       Hopefully the ranker will predict the same ranking for passages.'''

    test_ref_values = []
    for item in data:
        if criteria == 'acc-nli':
            test_ref_values.append([(float(ctx["acc"]) + float(ctx["NLI"]))/2 for ctx in item["ctxs"][:top_n]])          
        elif criteria == 'acc_LM-nli':
            test_ref_values.append([(float(ctx["acc_LM"]) + float(ctx["NLI"]))/2 for ctx in item["ctxs"][:top_n]])                
        elif criteria == 'nli':
            test_ref_values.append([float(ctx["NLI"]) for ctx in item["ctxs"][:top_n]])        
        elif criteria == 'acc':
            test_ref_values.append([float(ctx["acc"]) for ctx in item["ctxs"][:top_n]]) # not sure how much makes sense to eval rank w/ hard {0,1}       
        elif criteria == 'acc_LM':
            test_ref_values.append([float(ctx["acc_LM"]) for ctx in item["ctxs"][:top_n]]) # not sure how much makes sense to eval rank w/ hard {0,1}        
        else:
            print('Criteria for reference ranking unimplemented', criteria)
            exit(0)

    return test_ref_values

def construct_pairwise_dataset(data, top_n, single_net, criteria='acc_LM-nli', add_title=False):
    """
    Function for constructing a pairwise training set where each pair consists of a matching QA sequence and a
    non-matching QA sequence.
    :data: the loaded data from the input json file.
    :top_n: Number of passages to use to create pairs for a given question.
    :single_net: if this is true, create data without the pair-wise ranking. Not to change the dataset format, we just repeat the element in the tuple, e.g., (a,a).
    :acc_LM-nli: The criteria to use as score to build the (passage, question) pairs.
    :add_title: Whether to include the title concatenated with the passage or not.
    :return:
    """

    acc_type = 'acc_LM' if 'acc_LM' in criteria else 'acc'
    ctxq_pairs = []
    for i, item in enumerate(data):
        assert len(item["ctxs"]) >= 2
        if single_net:
            res = [(a, a) for idx, a in enumerate(item["ctxs"][:top_n])]
        else:
            res = [(a, b) for idx, a in enumerate(item["ctxs"][:top_n]) for b in item["ctxs"][:top_n][idx + 1:]]
        # we know that ctxs are sorted by retriever score
        for (a, b) in res:
            if criteria == 'acc-nli':
                a_score = (float(a["acc"]) + float(a["NLI"]))/2
                b_score = (float(b["acc"]) + float(b["NLI"]))/2
            elif criteria == 'acc_LM-nli':
                a_score = (float(a["acc_LM"]) + float(a["NLI"]))/2
                b_score = (float(b["acc_LM"]) + float(b["NLI"]))/2                   
            elif criteria == 'nli':
                a_score = float(a["NLI"])
                b_score = float(b["NLI"])                    
            elif criteria == 'acc':
                a_score = float(a["acc"])
                b_score = float(b["acc"])
            elif criteria == 'acc_LM':
                a_score = float(a["acc_LM"])
                b_score = float(b["acc_LM"])
            else:
                print('Criteria for reference ranking unimplemented', criteria)
                exit(0)

            # to concat the title we could just add a space, see here:
            # https://tinkerd.net/blog/machine-learning/bert-tokenization/
            if a_score > b_score:
                ctxq_best  = (item["question"], a["text"] if not add_title else a["title"] + "\n" + a["text"])
                ctxq_worse = (item["question"], b["text"] if not add_title else b["title"] + "\n" + b["text"])
                best_score = a_score
                worse_score = b_score
                best_acc = a[acc_type]
                worse_acc = b[acc_type]
            elif a_score == b_score and not single_net: 
                continue # if we are doing pair-wise ranking (siamese-net) and values are equal just skip the pair
            else:
                # Can come here because we have equal values. By construction duplicates when we want to use the 
                # network for single element evaluation will come here. Not efficient, not nice, but we reuse all same code.
                # The single net is used for when we train with only the BCE objective.
                ctxq_best  = (item["question"], b["text"] if not add_title else b["title"] + "\n" + b["text"])
                ctxq_worse = (item["question"], a["text"] if not add_title else a["title"] + "\n" + a["text"])
                best_score = b_score
                worse_score = a_score
                best_acc = b[acc_type]
                worse_acc = a[acc_type]
            ctxq_pairs.append((ctxq_best, ctxq_worse, best_score, worse_score, best_acc, worse_acc))

    return ctxq_pairs


def load_ragqa(input_file, top_n, criteria, interactive, shards=0, single_net=False, add_title=False): 
    
    if shards != 0:
        files = glob.glob(input_file.replace('.jsonl', '-sh*.jsonl'))
        assert shards == len(files), f'The number of specified shards ({shards}) should coincide with the number of files found, #files:{len(files)}.'
        input_data = []
        for i in range(shards):
            input_data.extend(load_file(input_file.replace('.jsonl', f'-sh{i+1}.jsonl')))
    else:
        input_data = load_file(input_file)
    print('File uploaded, ', input_file, len(input_data)) 

    if interactive:
        ref_values = construct_dataset(input_data, top_n, single_net, criteria=criteria)
        return input_data, ref_values
    else:
        return construct_pairwise_dataset(input_data, top_n, single_net, criteria=criteria, add_title=add_title)
    
    
