import torch

import argparse

parser = argparse.ArgumentParser(description="merge the replacement")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
parser.add_argument('--reranker', type=str, choices=[ 'Qwen3-Reranker-8B', 'ms-marco-MiniLM-L12', 'BGE-v2-m3'], required=True)

args = parser.parse_args()

shap_llama_list = torch.load(f'results/Llama-3.2-3B-Instruct_{args.dataset}/shap_{args.reranker}_collection.pt')
shap_gemma_replace = torch.load(f'results/gemma-3-4B-it_{args.dataset}/shap_{args.reranker}_replacement.pt')
tokenization_diff = torch.load(f"results/gemma-3-4B-it_{args.dataset}/tokenization_diff.pt")

shap_gemma_list = shap_llama_list.copy()

for i in range(len(shap_gemma_list)):
    if i in tokenization_diff:
        pos = list(tokenization_diff).index(i)
        shap_gemma_list[i] = shap_gemma_replace[pos]

torch.save(shap_gemma_list,f'results/gemma-3-4B-it_{args.dataset}/shap_{args.reranker}_collection.pt')

