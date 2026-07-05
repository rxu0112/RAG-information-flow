import argparse
import numpy as np
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import sys
from tqdm import tqdm 
from Other_Baselines.ragu.utils.utils import load_file, PROMPT_DICT, save_file_jsonl, postprocess_answers_closed
from Other_Baselines.ragu.retrieval_qa.metrics import metric_max_over_ground_truths, exact_match_score, match, f1


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def nliEval(model, tokenizer, premise_raw, hypothesis_raw):
    batch_tokens = tokenizer.batch_encode_plus(list(zip(premise_raw, hypothesis_raw)), padding=True, max_length=512, return_tensors="pt" ,truncation=True)
    with torch.no_grad():
        model_outputs = model(**{k: v.to(torch.cuda.current_device()) for k, v in batch_tokens.items()})
    batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
    batch_evids = batch_probs[:, 0].tolist() #entailment_idx
    #batch_conts = batch_probs[:, 1].tolist() #contradiction_idx
    #batch_neuts = batch_probs[:, 2].tolist() #neutral_idx
    #print(batch_evids)
    #exit(0)
    return batch_evids #, batch_neuts, batch_conts    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--nli_model', type=str, default="ragu/retrieval_qa/albert-xlarge-vitaminc-mnli")
    parser.add_argument('--ares_model', type=str, default=None)
    parser.add_argument('--input_file', type=str, default='baseline/ragu/new_data/hotpot/gemma_test_utility_distil_run_llm.jsonl')
    parser.add_argument('--result_fp', type=str, default='baseline/ragu/new_data/hotpot/gemma_test_distill_score.jsonl')    
    parser.add_argument('--top_n', type=int, default=5,
                        help="number of paragraphs to be considered.")
    
    args = parser.parse_args()

    input_data = load_file(args.input_file)


    if args.nli_model is not None:
        nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
        nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).eval()
        nli_model.to(torch.cuda.current_device())
        nli_model.half()  # use fp16 as in summac

        for item in tqdm(input_data, desc="Processing items", unit="item"):
            premise = []
            hypothesis = []
            for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                premise.append(ctx["text"])
                hypothesis.append(item["question"] + " " + ctx["output"])

            entail_results = nliEval(nli_model, nli_tokenizer, premise, hypothesis)
            for i, ctx in enumerate(item["ctxs"][:args.top_n]):
                ctx["NLI"] = entail_results[i]

    if args.ares_model is not None:
        print('Not implemented')
        exit(0)

    save_file_jsonl(input_data, args.result_fp)
    print('Files saved to', args.result_fp)


if __name__ == "__main__":
    main()