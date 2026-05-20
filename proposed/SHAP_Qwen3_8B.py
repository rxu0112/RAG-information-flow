import shap
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import argparse
import json

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

parser = argparse.ArgumentParser(description="SHAP Qwen3 8B")
parser.add_argument('--setup', type=str, choices=['Llama-3.2-3B-Instruct_squad2',
                                                  'Llama-3.2-3B-Instruct_hotpot', 
                                                  'Llama-3.2-3B-Instruct_msmarco', 
                                                  'gemma-3-4B-it_squad2',
                                                  'gemma-3-4B-it_hotpot',
                                                  'gemma-3-4B-it_msmarco'], required=True)
args = parser.parse_args()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if "Llama" in args.setup:
    prediction_path = f'results/{args.setup}/prediction_collection_bf16.json'
    save_path = f"results/{args.setup}/shap_Qwen3-Reranker-8B_collection.pt"

    with open(prediction_path, 'r') as f:
        prediction = json.load(f)
    try:
        results = torch.load(save_path,weights_only=False)
        begin = len(results)
        print(f"Loaded {begin} previous results from {save_path}")
    except FileNotFoundError:
        results = []
        begin = 0
        print("No previous results found. Starting fresh.")
else:
    prediction_path = f'results/{args.setup}/prediction_collection_bf16.json'
    save_path = f"results/{args.setup}/shap_Qwen3-Reranker-8B_replacement.pt"
    diff_path = f"results/{args.setup}/tokenization_diff.pt"
    with open(prediction_path, 'r') as f:
        prediction = json.load(f)
    tokenization_diff = torch.load(diff_path,weights_only=False)
    try:
        results = torch.load(save_path,weights_only=False)
        begin = len(results)
        print(f"Loaded {begin} previous results from {save_path}")
    except FileNotFoundError:
        results = []
        begin = 0
        print("No previous results found. Starting fresh.")
    

# Load your cross-encoder model
path = 'MODEL_PATH_PLACEHOLDER' # path of Qwen3-Reranker-8B
tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.float16).eval()
model.eval()
model.to(device)

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
task = 'Given a web search query, retrieve relevant passages that answer the query.'
def explain_example(question, context): # Predict function for SHAP: only takes masked contexts
    # call_counter = {"total": 0}
    def predict_context_only(masked_contexts):
        # call_counter["total"] += len(masked_contexts)
        # print("Batch:", len(masked_contexts), " | Total so far:", call_counter["total"])
        pairs = [format_instruction(task, question, c) for c in masked_contexts]
        inputs = process_inputs(pairs)
        scores = compute_logits(inputs)
        return np.array(scores).reshape(-1, 1)# Ensure shape (n_samples, 1) for SHAP
    
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_context_only, masker)
    
    shap_values = explainer([context])
    # print("Final total masks for this sample:", call_counter["total"])
    return shap_values

def is_word_or_number(token):
    return any(ch.isalnum() for ch in token)  # letters or digits


def find_section(tokens, start, keyword):
    matches = [i + start for i, t in enumerate(tokens[start:]) if t.strip() == keyword]
    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        return None
    return matches[0]

if vars().get('tokenization_diff'):
    prediction = [prediction[i] for i in tokenization_diff]

for i, example in enumerate(tqdm(prediction[begin:])):
    i = i+begin
    tokens = example['prompt_tokens']
    context_token_idx = find_section(tokens, 0, "Context")

    if context_token_idx is None:
        results.append([f"Skipping example {i}: 'Context' section not found."])
        continue

    question_token_idx = find_section(tokens, context_token_idx, "Question")

    if question_token_idx is None:
        results.append([f"Skipping example {i}: 'Question' section not found."])
        continue

    c_start = context_token_idx + 1
    while not is_word_or_number(tokens[c_start]):
        c_start += 1
    q_start = question_token_idx + 1
    while not is_word_or_number(tokens[q_start]):
        q_start += 1
    answer_token_idx = find_section(tokens, q_start, "Answer")
    
    if answer_token_idx is None:
        results.append([f"Skipping example {i}: 'Answer' section not found."])
        continue

    context = ''.join(tokens[c_start:question_token_idx])
    question = ''.join(tokens[q_start:answer_token_idx])
    if len(context.strip()) == 0 or len(question.strip()) == 0:
        results.append([f"Skipping example {i} due to empty context or question"])
    else:
        shap_values = explain_example(question, context)
        results.append(shap_values)

    if (i + 1) % 50 == 0:
        torch.save(results, save_path)

torch.save(results, save_path)