from transformers import AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="faithfulness evaluation")
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct','Llama-3-8B-Instruct','gemma-3-4B-it'], required=True)
parser.add_argument('--dataset', type=str, choices=['squad2','hotpot','msmarco'], required=True)

args = parser.parse_args()

with open(f'results/{args.model}_{args.dataset}/prediction_faithfulness_bf16.json', "r") as f:
    prediction = json.load(f)
save_path = f'results/{args.model}_{args.dataset}/hem_faithfulness_bf16.json'
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

path = 'MODEL_PATH_PLACEHOLDER' # path of hallucination_evaluation_model
model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True)
model.eval()
model.to(device)

results = []
for sample in tqdm(prediction):
    if sample['is_impossible']==False:
        merged_true_answers = sample['merged_true_answers']

        merged_prediction_wo_top_rank = sample['merged_prediction_wo_top_rank']
        scores_wo_top_rank = [model.predict([(ref, merged_prediction_wo_top_rank)])[0] for ref in merged_true_answers]
        hem_score_wo_top_rank = max(scores_wo_top_rank).item()

        merged_prediction_wo_low_rank = sample['merged_prediction_wo_low_rank']
        scores_wo_low_rank = [model.predict([(ref, merged_prediction_wo_low_rank)])[0] for ref in merged_true_answers]
        hem_score_wo_low_rank = max(scores_wo_low_rank).item()

        merged_prediction_wo_top_contri = sample['merged_prediction_wo_top_contri']
        scores_wo_top_contri = [model.predict([(ref, merged_prediction_wo_top_contri)])[0] for ref in merged_true_answers]
        hem_score_wo_top_contri = max(scores_wo_top_contri).item()

        merged_prediction_wo_low_contri = sample['merged_prediction_wo_low_contri']
        scores_wo_low_contri = [model.predict([(ref, merged_prediction_wo_low_contri)])[0] for ref in merged_true_answers]
        hem_score_wo_low_contri = max(scores_wo_low_contri).item()

        merged_prediction_wo_top_path = sample['merged_prediction_wo_top_path']
        scores_wo_top_path = [model.predict([(ref, merged_prediction_wo_top_path)])[0] for ref in merged_true_answers]
        hem_score_wo_top_path = max(scores_wo_top_path).item()

        merged_prediction_wo_low_path = sample['merged_prediction_wo_low_path']
        scores_wo_low_path = [model.predict([(ref, merged_prediction_wo_low_path)])[0] for ref in merged_true_answers]
        hem_score_wo_low_path = max(scores_wo_low_path).item()

        results.append([hem_score_wo_top_rank,hem_score_wo_low_rank,
                        hem_score_wo_top_contri,hem_score_wo_low_contri,
                        hem_score_wo_top_path,hem_score_wo_low_path])
    else:
        continue

with open(save_path, "w") as f:
    json.dump(results, f) 