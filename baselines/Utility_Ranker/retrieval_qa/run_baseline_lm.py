import argparse
import numpy as np
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import torch.multiprocessing as mp
import sys
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utility_Ranker.retrieval_qa.metrics import metric_max_over_ground_truths, exact_match_score, match, f1
from Utility_Ranker.utils.utils import load_file, PROMPT_DICT, save_file_jsonl, getChatMessages, call_model

# from lm_polygraph.estimators import FisherRao, RenyiNeg, MeanPointwiseMutualInformation, \
#                                 MeanConditionalPointwiseMutualInformation, MeanTokenEntropy
# from lm_polygraph.stat_calculators import EntropyCalculator

def split_dataset(dataset):
    """Get indices of answerable and unanswerable questions."""

    def clen(ex):
        if type(ex["answers"]) is list:
            return len(ex["answers"])
        elif type(ex["answers"]) is dict:
            return len(ex["answers"]["text"])
        else:
            return len(ex["answers"])

    answerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) > 0]
    unanswerable_indices = [i for i, ex in enumerate(dataset) if clen(ex) == 0]

    # union == full dataset
    assert set(answerable_indices) | set(
        unanswerable_indices) == set(range(len(dataset)))
    # no overlap
    assert set(answerable_indices) - \
        set(unanswerable_indices) == set(answerable_indices)

    return answerable_indices, unanswerable_indices

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="llama3-8B")
    parser.add_argument(
        '--input_file',
        type=str,
        default="",
    )
    parser.add_argument('--top_n', type=int, default=5,
                        help="number of paragraphs to be considered.")
    parser.add_argument(
        '--result_fp',
        type=str,
        default="",
    )
    parser.add_argument('--prompt_name', type=str, default="chat_directRagQA_REAR2")
    parser.add_argument("--instruction",  type=str,
                        default=None, help="task instructions") # REMOVE THIS EVERYWHERE
    parser.add_argument('--chat_template', action="store_true")
    parser.add_argument('--fewshots', type=str, help="list of examples used for few-shot prompting --jsonl file",
                        default=None) 
    parser.add_argument('--split', type=str, help="split to process", default=None) 
    parser.add_argument('--compute_pmi', action="store_true")
    parser.add_argument('--sort_ctx', action="store_true")
    parser.add_argument('--sort_ctx_criteria', type=str, default="acc_LM-nli_pred")
    parser.add_argument('--top_n_to_rank', type=int, default=10,
                        help="number of paragraphs to be considered for re-ranking.")   
    parser.add_argument('--random_seed', type=int, default=10)                        
    parser.add_argument('--p_true_sample', action="store_true")
    parser.add_argument("--p_true_num_fewshot", type=int, default=20,
            help="Number of few shot examples to use")    
    parser.add_argument('--proportion', type=float, default=1.0,
            metavar='N', help='the proportion of data used')               
    
    # sampling params
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="temperature at decoding. Zero means greedy sampling.")                        
    parser.add_argument('--top_p', type=float, default=1.0,
                        help="top-p sampling.")
    parser.add_argument('--top_k', type=int, default=-1,
                    help="top-k sampling.")
    parser.add_argument('--logprobs', type=int, default=2,
                    help="number of log probs to return.")
    parser.add_argument('--do_stop', action="store_true", default=False)

    ## see other parameters                        
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=8,
                        help="world size to use multiple GPUs.")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--quantization', type=str, default=None)
  
    args = parser.parse_args()

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value

    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value

    random.seed(args.random_seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(args.random_seed)

    #Fix torch random seed
    #torch.manual_seed(args.random_seed)


    
    if args.dtype is not None:
        model = LLM(model=args.model_name, dtype=args.dtype,
                    tensor_parallel_size=args.world_size, max_logprobs=args.logprobs + 1) #download_dir=args.download_dir,
    elif 'Llama-3.1-70B' in args.model_name and 'AWQ' in args.model_name:
        # https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html
        model = LLM(model=args.model_name, quantization='awq',
                    tensor_parallel_size=args.world_size, trust_remote_code=True,
                    gpu_memory_utilization=0.9, max_model_len=14336, enforce_eager=False) #download_dir=args.download_dir, awq_marlin 
    else:
        model = LLM(model=args.model_name, 
                    tensor_parallel_size=args.world_size, max_logprobs=args.logprobs + 1,
                    trust_remote_code=True, gpu_memory_utilization=0.9,) # download_dir=args.download_dir,
    tokenizer = model.get_tokenizer()

    input_data = load_file(args.input_file)
    print('File uploaded, ', args.input_file, len(input_data))

    _prompt_name = args.prompt_name
    if 'Llama' in args.model_name:
        _prompt_name += 'Llama'


    # if using few shot training examples, remove them from the training, create the prompt string
    if args.fewshots:
        fewshots = load_file(args.fewshots)
        if args.split == 'train':
            fewshots_ids = [x['q_id'] for x in fewshots]
            input_data = [x for x in input_data if not x['q_id'] in fewshots_ids]
            print('Fewshots removed, ', len(input_data))
        fewshot_prompt = "".join(['{}\nAnswer: {}\n\n'.format(x['question'],x['answers'][0]) for x in fewshots])

    if args.split == 'train' and args.p_true_sample:
        print(f'Generating Most Likely Answer for each of the {args.p_true_num_fewshot} p(true) ICL exmaples (taking the shots from train proportion {args.proportion}).')
        input_data = input_data[:int(len(input_data)*args.proportion)]
        # Get indices of answerable and unanswerable questions and construct prompt.
        answerable_indices, unanswerable_indices = split_dataset(input_data) 
        p_true_indices = random.sample(answerable_indices, args.p_true_num_fewshot)
        input_data = [x for i, x in enumerate(input_data) if i in p_true_indices] # take the sampled train examples    

    for id, item in enumerate(input_data):
        if args.sort_ctx and args.sort_ctx_criteria in item["ctxs"][0].keys():
            retrieval_result = item["ctxs"][:args.top_n_to_rank]
            #retrieval_result = sorted(retrieval_result, key=itemgetter(args.sort_ctx_criteria), reverse=True)
            retrieval_result = sorted(retrieval_result, key=lambda d: d[args.sort_ctx_criteria], reverse=True)
            retrieval_result = retrieval_result[:args.top_n]
        else:
            retrieval_result = item["ctxs"][:args.top_n]
        # evidences = ["[{}] ".format(
        #     i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
        evidences = ["[{}] ".format(
            i+1) + ctx["text"] for i, ctx in enumerate(retrieval_result)]
        # item["paragraph"] = "\n".join(evidences)
        item["context"] = "\n".join(evidences)
        del item["ctxs"] # remove original ctxs to form the output dict, will add prompt below

        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]
            print(item["instruction"] + '\n')

        if args.fewshots:
            item['fewshots'] = fewshot_prompt
            

    # entropy = EntropyCalculator()
    # fisher_rao = FisherRao()
    # renyi_neg = RenyiNeg()
    # mean_pmi = MeanPointwiseMutualInformation()
    # mean_cond_pmi = MeanConditionalPointwiseMutualInformation()
    # mte = MeanTokenEntropy()

    # debug
    #input_data = input_data[:20]

    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        if args.chat_template:
            processed_batch = []
            for item in batch:
                # By the time I'm writting this vLLM does not have the chat_template creation so we need to manually
                # do it. But concurently it seems that they are working on the vLLM to include this automatically?
                # see here: https://github.com/vllm-project/vllm/pull/6936
                # so keep to the current version w/ this code or watch out not to apply twice!!
                #https://huggingface.co/docs/transformers/chat_templating
                #https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py
                #documents = []
                #for ctx in item["ctxs"][:args.top_n]:
                #    documents.append({"title": ctx["title"],
                #                      "content": ctx["text"]})
                tokenized_chat = tokenizer.apply_chat_template(
                                    getChatMessages(args.model_name, _prompt_name, item), 
                                    tokenize=False, add_generation_prompt=True)
                processed_batch.append(tokenized_chat)

        else:
            processed_batch = [
                PROMPT_DICT[_prompt_name]['user'].format_map(item) for item in batch]
        for item in batch:
            del item["context"] # just use this dict-key to re-use format
            del item["instruction"] # just use this dict-key to re-use format

        preds, toklogprobs, sectoklogprob, toklogdists_pred, toklogprobs_lm = call_model(processed_batch, model, args, tokenizer)

        for j, item in enumerate(batch):
            item["prompt"] = processed_batch[j]
            item["output"] = preds[j]
            item["toklogprob"] = toklogprobs[j]
            item["2ndtoklogprob"] = sectoklogprob[j]
            # entropies = entropy({'greedy_log_probs': np.array([toklogdists_pred[j]])})["entropy"]
            # item["entropy"] = entropies[0]
            if args.compute_pmi:
                item["toklogprobs_lm"] = toklogprobs_lm[j]
            if not toklogdists_pred[j]:
                print(item["prompt"])
                print(item["output"])
                print(toklogdists_pred[j])
            # uncertainty estimators
            stats = {
                'greedy_log_probs': np.array([toklogdists_pred[j]]),
                'greedy_log_likelihoods': np.array([toklogprobs[j]]),
                # 'entropy': entropies,
                # 'greedy_lm_log_likelihoods': np.array([toklogprobs_lm[j]])
            }
            # item["FisherRao"] = fisher_rao(stats)[0] # higher more uncertain
            # item["RenyiNeg"] = renyi_neg(stats)[0] # higher more uncertain
            # item["PMI"] = mean_pmi(stats)[0] # higher more uncertain  
            # item["condPMI"] = mean_cond_pmi(stats)[0] # higher more uncertain  
            # item["MeanTokenEntropy"] = mte(stats)[0] # higher more uncertain

            item["nll"] = (-sum(item["toklogprob"]))/len(item["toklogprob"]) if len(item["toklogprob"])>0 else 0
            item["ppl"] = np.exp(item["nll"])
            item["MSP"] = -sum(item["toklogprob"]) # as in Polygraph, higher values, higher uncertainty
           

    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx+1)*args.batch_size:]
        if args.chat_template:
            processed_batch = []
            for item in batch:
                #https://huggingface.co/docs/transformers/chat_templating
                #https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py
                #documents = []
                #for ctx in item["ctxs"][:args.top_n]:
                #    documents.append({"title": ctx["title"],
                #                      "content": ctx["text"]})
                tokenized_chat = tokenizer.apply_chat_template(
                                    getChatMessages(args.model_name, _prompt_name, item), 
                                    tokenize=False, add_generation_prompt=True)
                processed_batch.append(tokenized_chat)

        else:
            processed_batch = [
                PROMPT_DICT[_prompt_name]['user'].format_map(item) for item in batch]
        
        preds, toklogprobs, sectoklogprob, toklogdists_pred, toklogprobs_lm = call_model(processed_batch, model, args, tokenizer)

        for j, item in enumerate(batch):
            item["prompt"] = processed_batch[j]
            item["output"] = preds[j]
            item["toklogprob"] = toklogprobs[j]
            item["2ndtoklogprob"] = sectoklogprob[j]
            # entropies = entropy({'greedy_log_probs': np.array([toklogdists_pred[j]])})["entropy"]
            # item["entropy"] = entropies[0]
            if args.compute_pmi:
                item["toklogprobs_lm"] = toklogprobs_lm[j]
            # uncertainty estimators
            # stats = {
            #     'greedy_log_probs': np.array([toklogdists_pred[j]]),
            #     'greedy_log_likelihoods': np.array([toklogprobs[j]]),
            #     'entropy': entropies,
            #     'greedy_lm_log_likelihoods': np.array([toklogprobs_lm[j]])
            # }
            # item["FisherRao"] = fisher_rao(stats)[0] # higher more uncertain
            # item["RenyiNeg"] = renyi_neg(stats)[0] # higher more uncertain
            # item["PMI"] = mean_pmi(stats)[0] # higher more uncertain  
            # item["condPMI"] = mean_cond_pmi(stats)[0] # higher more uncertain  
            # item["MeanTokenEntropy"] = mte(stats)[0] # higher more uncertain

            item["nll"] = (-sum(item["toklogprob"]))/len(item["toklogprob"]) if len(item["toklogprob"])>0 else 0
            item["ppl"] = np.exp(item["nll"])
            item["MSP"] = -sum(item["toklogprob"]) # as in Polygraph, higher values, higher uncertainty /error

    print('Finalised QA, start evaluation...')
    #rouge = evaluate.load('rouge')

    for item in input_data:
        # peformance
        item["EM"] = metric_max_over_ground_truths(
                exact_match_score, item["output"], item["golds"])
        item["F1"] = f1([item["output"]], item["golds"])
        item["acc"] = match(item["output"], item["golds"])
        item["rougeL"] = 0 #max([rouge.compute(predictions=[item["output"]], references=[answer])["rougeL"]
                            #for answer in item["golds"]]) ## no need .mid.fmeasure as using aggregator and returning fmeasure

    print("overall result EM: {0}".format(
        np.mean([item["EM"] for item in input_data])))
    print("overall result F1: {0}".format(
        np.mean([item["F1"] for item in input_data])))
    print("overall result acc: {0}".format(
        np.mean([item["acc"] for item in input_data])))               
    #print("overall result rougeL: {0}".format(
    #    np.mean([item["rougeL"] for item in input_data]))) 
    print("overall result ppl: {0}".format(
        np.mean([item["ppl"] for item in input_data])))         


    save_file_jsonl(input_data, args.result_fp)
    print('Files saved to, ', args.result_fp)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
