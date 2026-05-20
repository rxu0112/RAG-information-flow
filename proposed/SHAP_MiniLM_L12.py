import shap
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
import argparse
import json

parser = argparse.ArgumentParser(description="SHAP MiniLM")
parser.add_argument('--setup', type=str, choices=['Llama-3.2-3B-Instruct_squad2',
                                                  'Llama-3.2-3B-Instruct_hotpot',
                                                  'Llama-3.2-3B-Instruct_msmarco',
                                                  'gemma-3-4B-it_squad2',
                                                  'gemma-3-4B-it_hotpot',
                                                  'gemma-3-4B-it_msmarco'], required=True)
args = parser.parse_args()
device = 'cuda:6' if torch.cuda.is_available() else 'cpu'

if "Llama" in args.setup:
    prediction_path = f'results/{args.setup}/prediction_collection_bf16.json'
    save_path = f"results/{args.setup}/shap_ms-marco-MiniLM-L12_collection.pt"

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
    save_path = f"results/{args.setup}/shap_ms-marco-MiniLM-L12_replacement.pt"
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
path = 'MODEL_PATH_PLACEHOLDER' # path of ms-marco-MiniLM-L12-v2
model = CrossEncoder(path)
tokenizer = AutoTokenizer.from_pretrained(path)
model.eval()
model.to(device)

def explain_example(question, context): # Predict function for SHAP: only takes masked contexts
    # call_counter = {"total": 0}
    def predict_context_only(masked_contexts):
        # call_counter["total"] += len(masked_contexts)
        # print("Batch:", len(masked_contexts), " | Total so far:", call_counter["total"])
        pairs = [(question, c) for c in masked_contexts]
        scores = model.predict(pairs)
        return scores.reshape(-1, 1) # Ensure shape (n_samples, 1) for SHAP
    
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