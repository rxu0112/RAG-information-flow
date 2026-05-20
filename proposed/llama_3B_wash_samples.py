import json
from utils import *
import torch
import argparse

parser = argparse.ArgumentParser(description="remove low-quality samples")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
args = parser.parse_args()


with open(f'results/Llama-3.2-3B-Instruct_{args.dataset}/prediction_collection_bf16.json', 'r') as f:
    prediction_list = json.load(f)
shap_ms_list = torch.load(f'results/Llama-3.2-3B-Instruct_{args.dataset}/shap_ms-marco-MiniLM-L12_collection.pt')
shap_qwen_list = torch.load(f'results/Llama-3.2-3B-Instruct_{args.dataset}/shap_Qwen3-Reranker-8B_collection.pt')
shap_BGE_list = torch.load(f'results/Llama-3.2-3B-Instruct_{args.dataset}/shap_BGE-v2-m3_collection.pt')

indices_to_remove = []
full_stop_positions = []

for s_idx, pred in enumerate(prediction_list):
    tokens = pred['generated_tokens']
    dot_positions = [i for i, tok in enumerate(tokens) if "." in tok]
    if dot_positions:
        pos = dot_positions[0]  # first token containing "."
    else:
        pos = len(tokens) - 1  # last position if no dot
    full_stop_positions.append(pos)

    if pos == 0 or pos == 9:
        indices_to_remove.append(s_idx)
    if pos != len(tokens)-2: # remove samples if not end with dot
        indices_to_remove.append(s_idx)

    if pred['true_answers']:
        true_text = pred['true_answers'][0].strip()
        if "." in true_text[0:-1]:
            indices_to_remove.append(s_idx)

start_list = []
end_list = []
for s_idx, pred in enumerate(prediction_list):
    context_token_idx = find_section(pred['prompt_tokens'], 0, "Context")

    if context_token_idx is None:
        print(f"Skipping example {s_idx}: 'Context' section not found.")
        indices_to_remove.append(s_idx)
        start_list.append(None)
        end_list.append(None)
        continue

    question_token_idx = find_section(pred['prompt_tokens'], context_token_idx, "Question")
    if question_token_idx is None:
        print(f"Skipping example {s_idx}: 'Question' section not found.")
        indices_to_remove.append(s_idx)
        start_list.append(None)
        end_list.append(None)
        continue

    c_start = context_token_idx + 1
    while not is_word_or_number(pred['prompt_tokens'][c_start]):
        c_start += 1
    q_start = question_token_idx + 1
    while not is_word_or_number(pred['prompt_tokens'][q_start]):
        q_start += 1
    answer_token_idx = find_section(pred['prompt_tokens'], q_start, "Answer")

    if answer_token_idx is None:
        print(f"Skipping example {s_idx}: 'Answer' section not found.")
        indices_to_remove.append(s_idx)
        start_list.append(None)
        end_list.append(None)
        continue

    context = ''.join(pred['prompt_tokens'][c_start:question_token_idx])
    question = ''.join(pred['prompt_tokens'][q_start:answer_token_idx])

    if len(context.strip()) == 0 or len(question.strip()) == 0:
        print(f"Skipping example {s_idx} due to empty context or question")
        indices_to_remove.append(s_idx)
        start_list.append(None)
        end_list.append(None)
        continue
        
    start_list.append(c_start)
    end_list.append(question_token_idx)

for shap_list in [shap_qwen_list, shap_ms_list, shap_BGE_list]:
    for s_idx, pred in enumerate(prediction_list):
        if s_idx not in indices_to_remove:
            tokens = shap_list[s_idx][0][:, 0].data
            if tokens[-1] == '':
                tokens = tokens[0:-1]
            context_tokens = np.array(pred['prompt_tokens'][start_list[s_idx]:end_list[s_idx]])
            merged_context_tokens = "".join(context_tokens)
            merged_shap_tokens = "".join(tokens)
            if merged_context_tokens != merged_shap_tokens:
                indices_to_remove.append(s_idx)

indices_to_remove = sorted(set(indices_to_remove))

torch.save(indices_to_remove, f"results/Llama-3.2-3B-Instruct_{args.dataset}/indices_to_remove.pt")
torch.save(full_stop_positions, f"results/Llama-3.2-3B-Instruct_{args.dataset}/full_stop_positions.pt")



