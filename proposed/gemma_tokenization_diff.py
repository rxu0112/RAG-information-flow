import json
import torch
from utils import *
import argparse

parser = argparse.ArgumentParser(description="find different tokenization between LLM models")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
args = parser.parse_args()

with open(f'results/Llama-3.2-3B-Instruct_{args.dataset}/prediction_collection_bf16.json','r') as f:
    llama_pred = json.load(f)

with open(f'results/gemma-3-4B-it_{args.dataset}/prediction_collection_bf16.json','r') as f:
    gemma_pred = json.load(f)

llama_context = []
llama_question = []
llama_skipped = []
for i, example in enumerate(llama_pred):
    tokens = example['prompt_tokens']
    context_token_idx = find_section(tokens, 0, "Context")
    if context_token_idx is None:
        print(f"Skipping example {i}: 'Context' section not found.")
        llama_context.append([f"Skipping example {i}: 'Context' section not found."])
        llama_question.append([f"Skipping example {i}: 'Context' section not found."])
        llama_skipped.append(i)
        continue
    question_token_idx = find_section(tokens, context_token_idx, "Question")
    if question_token_idx is None:
        print(f"Skipping example {i}: 'Question' section not found.")
        llama_context.append([f"Skipping example {i}: 'Question' section not found."])
        llama_question.append([f"Skipping example {i}: 'Question' section not found."])
        llama_skipped.append(i)
        continue
    c_start = context_token_idx + 1
    while not is_word_or_number(tokens[c_start]):
        c_start += 1
    q_start = question_token_idx + 1
    while not is_word_or_number(tokens[q_start]):
        q_start += 1
    answer_token_idx = find_section(tokens, q_start, "Answer")
    if answer_token_idx is None:
        print(f"Skipping example {i}: 'Answer' section not found.")
        llama_context.append([f"Skipping example {i}: 'Answer' section not found."])
        llama_question.append([f"Skipping example {i}: 'Answer' section not found."])
        llama_skipped.append(i)
        continue
    context = ''.join(tokens[c_start:question_token_idx])
    question = ''.join(tokens[q_start:answer_token_idx])
    llama_context.append(context)
    llama_question.append(question)

gemma_context = []
gemma_question = []
gemma_skipped = []
for i, example in enumerate(gemma_pred):
    tokens = example['prompt_tokens']
    context_token_idx = find_section(tokens, 0, "Context")
    if context_token_idx is None:
        print(f"Skipping example {i}: 'Context' section not found.")
        gemma_context.append([f"Skipping example {i}: 'Context' section not found."])
        gemma_question.append([f"Skipping example {i}: 'Context' section not found."])
        gemma_skipped.append(i)
        continue
    question_token_idx = find_section(tokens, context_token_idx, "Question")
    if question_token_idx is None:
        print(f"Skipping example {i}: 'Question' section not found.")
        gemma_context.append([f"Skipping example {i}: 'Question' section not found."])
        gemma_question.append([f"Skipping example {i}: 'Question' section not found."])
        gemma_skipped.append(i)
        continue
    c_start = context_token_idx + 1
    while not is_word_or_number(tokens[c_start]):
        c_start += 1
    q_start = question_token_idx + 1
    while not is_word_or_number(tokens[q_start]):
        q_start += 1
    answer_token_idx = find_section(tokens, q_start, "Answer")
    if answer_token_idx is None:
        print(f"Skipping example {i}: 'Answer' section not found.")
        gemma_context.append([f"Skipping example {i}: 'Answer' section not found."])
        gemma_question.append([f"Skipping example {i}: 'Answer' section not found."])
        gemma_skipped.append(i)
        continue
    context = ''.join(tokens[c_start:question_token_idx])
    question = ''.join(tokens[q_start:answer_token_idx])
    gemma_context.append(context)
    gemma_question.append(question)

diff_context = []
for i in range(len(llama_pred)):
    if llama_context[i] != gemma_context[i]:
        diff_context.append(i)

diff_question = []
for i in range(len(llama_pred)):
    if llama_question[i] != gemma_question[i]:
        diff_question.append(i)

total_diff = sorted(set(diff_context + diff_question))

torch.save(total_diff,f"results/gemma-3-4B-it_{args.dataset}/tokenization_diff.pt")