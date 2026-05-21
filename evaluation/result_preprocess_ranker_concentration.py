import json
import torch
from utils import *
from rbo import rbo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)
parser.add_argument('--reranker', type=str, choices=['Qwen3' 'MiniLM','BGE'], required=True)
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct', 'gemma-3-4b-it'], required=True)
parser.add_argument('--p', type=float, required=True)

args = parser.parse_args()

device_num = 3

ranking_list = []
contri_list = []
path_list = []
for i in range(device_num):
    with open(
            f'results/{args.model}_{args.dataset}/loop_bf16/loop_manhattan_rank_bf16_{i}.jsonl',
            'r', encoding='utf-8') as f:
        rank = [json.loads(line) for line in f if line.strip()]
    with open(
            f'results/{args.model}_{args.dataset}/loop_bf16/loop_manhattan_contri_bf16_{i}.jsonl',
            'r', encoding='utf-8') as f:
        contri = [json.loads(line) for line in f if line.strip()]
    with open(
            f'results/{args.model}_{args.dataset}/loop_bf16/loop_manhattan_path_bf16_{i}.jsonl',
            'r', encoding='utf-8') as f:
        path = [json.loads(line) for line in f if line.strip()]
    ranking_list = ranking_list + rank
    contri_list = contri_list + contri
    path_list = path_list + path

with open(f'results/{args.model}_{args.dataset}/prediction_collection_bf16.json', 'r') as f:
    prediction_list = json.load(f)

with open(f'results/{args.model}_{args.dataset}/hem_answerable_collection_bf16.json', 'r') as f:
    hem_list = json.load(f)

if args.reranker == 'Qwen3':
    shap_list = torch.load(f'results/{args.model}_{args.dataset}/shap_Qwen3-Reranker-8B_collection.pt', weights_only=False)
elif args.reranker == 'MiniLM':
    shap_list = torch.load(f'results/{args.model}_{args.dataset}/shap_ms-marco-MiniLM-L12_collection.pt', weights_only=False)
elif args.reranker == 'BGE':
    shap_list = torch.load(f'results/{args.model}_{args.dataset}/shap_BGE-v2-m3_collection.pt', weights_only=False)

full_stop_positions = torch.load(f'results/{args.model}_{args.dataset}/full_stop_positions.pt', weights_only=False)
indices_to_remove = torch.load(f'results/{args.model}_{args.dataset}/indices_to_remove.pt', weights_only=False)

obtained_num = min([len(ranking_list), len(contri_list), len(path_list), len(prediction_list)])
prediction_list = prediction_list[0: obtained_num]
hem_list = hem_list[0: obtained_num]
shap_list = shap_list[0: obtained_num]
full_stop_positions = full_stop_positions[0: obtained_num]
ranking_list = [i['rankings'] for i in ranking_list[0:obtained_num]]
contri_list = [i['contri'] for i in contri_list[0:obtained_num]]
path_list = [i['path'] for i in path_list[0:obtained_num]]

prediction_list = [v for i, v in enumerate(prediction_list) if i not in indices_to_remove]
hem_list = [v for i, v in enumerate(hem_list) if i not in indices_to_remove]
shap_list = [v for i, v in enumerate(shap_list) if i not in indices_to_remove]
full_stop_positions = [v for i, v in enumerate(full_stop_positions) if i not in indices_to_remove]
contri_list = [v for i, v in enumerate(contri_list) if i not in indices_to_remove]
path_list = [v for i, v in enumerate(path_list) if i not in indices_to_remove]
ranking_list = [v for i, v in enumerate(ranking_list) if i not in indices_to_remove]

for s_idx, pos in enumerate(full_stop_positions):
    contri_list[s_idx] = contri_list[s_idx][0:pos]  # 'pos' exclude full stop; 'pos+1' include full stop
    path_list[s_idx] = path_list[s_idx][0:pos]
    ranking_list[s_idx] = ranking_list[s_idx][0:pos]

start_list = []
end_list = []
for s_idx, pred in enumerate(prediction_list):
    context_token_idx = find_section(pred['prompt_tokens'], 0, "Context")
    question_token_idx = find_section(pred['prompt_tokens'], context_token_idx, "Question")
    c_start = context_token_idx + 1
    while not is_word_or_number(pred['prompt_tokens'][c_start]):
        c_start += 1
    start_list.append(c_start)
    end_list.append(question_token_idx)

rank_score_mean_list = []
rank_score_max_list = []
for i, rankings in enumerate(ranking_list):
    score_list = []
    for t in rankings:
        score = np.empty_like(t).copy()
        score[t] = np.arange(len(t), 0, -1)
        score_list.append(to_simplex_shift(score[start_list[i]:end_list[i]]))
    arr = np.array(score_list, dtype=object)
    mean_vals = arr.mean(axis=0)
    max_vals = arr.max(axis=0)
    rank_score_mean_list.append(mean_vals)
    rank_score_max_list.append(max_vals)

rank_score_list = rank_score_max_list

contri_mean_list = []
contri_max_list = []
contri_min_list = []
for i, contri in enumerate(contri_list):
    score_list = []
    for score in contri:
        score_list.append(to_simplex_shift(score[start_list[i]:end_list[i]]))
    arr = np.array(score_list, dtype=object)
    mean_vals = arr.mean(axis=0)
    max_vals = arr.max(axis=0)
    min_vals = arr.min(axis=0)
    contri_mean_list.append(mean_vals)
    contri_max_list.append(max_vals)
    contri_min_list.append(min_vals)

contri_list = contri_max_list

path_mean_list = []
path_max_list = []
path_min_list = []

for i, path in enumerate(path_list):
    score_list = []
    for score in path:
        score_list.append(to_simplex_shift(score[start_list[i]:end_list[i]]))
    arr = np.array(score_list, dtype=object)
    mean_vals = arr.mean(axis=0)
    max_vals = arr.max(axis=0)
    min_vals = arr.min(axis=0)
    path_mean_list.append(mean_vals)
    path_max_list.append(max_vals)
    path_min_list.append(min_vals)

path_list = path_max_list

answerable_list = []
for shap in shap_list:
    answerable_list.append(np.sum(shap[0][:, 0].values))

shap_token_phrase = []
shap_value_phrase = []
for shap in shap_list:
    values, clustering = unpack_shap_explanation_contents(shap[0][:, 0])
    tokens, values, group_sizes = process_shap_values(shap[0][:, 0].data, values, grouping_threshold=0.01,
                                                    separator='', clustering=clustering)
    values = to_simplex_shift(values)

    if tokens[-1] == '':
        tokens = tokens[0:-1]
        values[-2] = values[-1] + values[-2]
        values = values[0:-1]

    shap_token_phrase.append(tokens)
    shap_value_phrase.append(values)

shap_token_word = []
shap_value_word = []
for shap in shap_list:
    tokens, values = merge_tokens_with_values(shap[0][:, 0].data, to_simplex_shift(shap[0][:, 0].values))

    if tokens[-1] == '':
        tokens = tokens[0:-1]
        values[-2] = values[-1] + values[-2]
        values = values[0:-1]

    shap_token_word.append(tokens)
    shap_value_word.append(values)

shap_token_initial = []
shap_value_initial = []
for shap in shap_list:
    tokens = shap[0][:, 0].data
    values = to_simplex_shift(shap[0][:, 0].values)

    if tokens[-1] == '':
        tokens = tokens[0:-1]
        values[-2] = values[-1] + values[-2]
        values = values[0:-1]

    shap_token_initial.append(tokens)
    shap_value_initial.append(values)

all_subword_rank_initial = []
all_subword_contri_initial = []
all_subword_path_initial = []
all_subword_tokens_initial = []
all_piece_imp_initial = []
all_piece_tokens_initial = []

for s_idx, pred in enumerate(prediction_list):
    print(s_idx)
    context_tokens = np.array(pred['prompt_tokens'][start_list[s_idx]:end_list[s_idx]])

    rank_values = to_simplex_shift(rank_score_list[s_idx])
    contri_values = to_simplex_shift(contri_list[s_idx])
    path_values = to_simplex_shift(path_list[s_idx])

    subword_rank_initial = []
    subword_contri_initial = []
    subword_path_initial = []
    subword_tokens_initial = []
    piece_imp_initial = []
    piece_tokens_initial = []

    subword_pos = 0
    piece_pos = 0

    while piece_pos < len(shap_token_initial[s_idx]):
        piece_buffer = shap_token_initial[s_idx][piece_pos]
        piece_imp = shap_value_initial[s_idx][piece_pos]
        subword_buffer = ""
        subword_contri = 0
        subword_path = 0
        subword_rank = []

        while subword_pos < len(context_tokens) and piece_pos < len(shap_token_initial[s_idx]):
            subword_buffer += context_tokens[subword_pos]
            subword_contri += contri_values[subword_pos]
            subword_rank.append(rank_values[subword_pos])
            subword_path += path_values[subword_pos]
            subword_pos += 1

            # Keep extending piece_buffer until lengths are compatible
            while len(piece_buffer.strip()) < len(subword_buffer.strip()):
                piece_pos += 1
                piece_buffer += shap_token_initial[s_idx][piece_pos]

            if subword_buffer.strip() == piece_buffer.strip():
                subword_rank_initial.append(np.mean(subword_rank))
                subword_contri_initial.append(subword_contri)
                subword_path_initial.append(subword_path)
                subword_tokens_initial.append(subword_buffer)
                piece_imp_initial.append(piece_imp)
                piece_tokens_initial.append(piece_buffer)
                piece_pos += 1
                break

    all_subword_rank_initial.append(subword_rank_initial)
    all_subword_contri_initial.append(subword_contri_initial)
    all_subword_tokens_initial.append(subword_tokens_initial)
    all_subword_path_initial.append(subword_path_initial)
    all_piece_imp_initial.append(piece_imp_initial)
    all_piece_tokens_initial.append(piece_tokens_initial)



KL_relevance = []
for s_idx in range(obtained_num-len(indices_to_remove)):
    KL_relevance.append(KL_divergence_with_uniform(all_piece_imp_initial[s_idx]))

labels = [1 if s > 0.5 else 0 for s in hem_list]
y = np.array(labels)

X = np.column_stack([KL_relevance])

scaler = StandardScaler()
X = scaler.fit_transform(X)
indices = np.arange(len(y))

temp_idx, test_idx, y_temp, y_test = train_test_split(
    indices, y, test_size=0.2)

train_idx, val_idx, y_train, y_val = train_test_split(
    temp_idx, y_temp, test_size=0.25)

X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

hem_train = np.array(hem_list)[train_idx]
hem_val = np.array(hem_list)[val_idx]
hem_test = np.array(hem_list)[test_idx]

test_summary = {"hem": hem_test, "X": X_test, "y": y_test}
train_summary = {"hem": hem_train, "X": X_train, "y": y_train}
val_summary = {"hem": hem_val, "X": X_val, "y": y_val}

torch.save(test_summary, f"results/{args.model}_{args.dataset}/ranker_concentration/{args.dataset}_test_uq_{args.reranker}_ranker_concentration.pt")
torch.save(train_summary, f"results/{args.model}_{args.dataset}/ranker_concentration/{args.dataset}_train_uq_{args.reranker}_ranker_concentration.pt")
torch.save(val_summary, f"results/{args.model}_{args.dataset}/ranker_concentration/{args.dataset}_val_uq_{args.reranker}_ranker_concentration.pt")