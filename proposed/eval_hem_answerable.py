from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Inference")
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct','gemma-3-4B-it', 'Llama-3-8B-Instruct'], required=True)
parser.add_argument('--dataset', type=str, choices=['squad2','hotpot','msmarco'], required=True)

args = parser.parse_args()

with open(f'results/{args.model}_{args.dataset}/prediction_collection_bf16.json', "r") as f:
    prediction = json.load(f)
save_path = f'results/{args.model}_{args.dataset}/hem_answerable_collection_bf16.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu' # specificy your device

tokenizer_path = 'MODEL_PATH_PLACEHOLDER' # path of flan-t5-base
tokenizer=AutoTokenizer.from_pretrained('...')
hem_path = 'MODEL_PATH_PLACEHOLDER' # path of hallucination_evaluation_model
model = AutoModelForSequenceClassification.from_pretrained(hem_path, trust_remote_code=True)
model.eval()
model.to(device)

results = []
for sample in tqdm(prediction):
    if sample['is_impossible']==False:
        merged_true_answers = sample['merged_true_answers']
        merged_prediction = sample['merged_prediction']
        scores = [model.predict([(ref, merged_prediction)])[0] for ref in merged_true_answers]
        hem_score = max(scores).item()
        results.append(hem_score)
    else:
        true_answers = ['I do not know.', 'It is not mentioned.','No one.','Nothing.','Answer is not available.','It is not explained.']
        pred_answer = sample['predicted_answer']
        scores = [model.predict([(ref, pred_answer)])[0] for ref in true_answers]
        hem_score = max(scores).item()
        results.append(hem_score)

with open(save_path, "w") as f:
    json.dump(results, f) 