import json
import torch
from utils import *
from rbo import rbo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, roc_curve, auc, precision_recall_curve
import argparse

parser = argparse.ArgumentParser(description="comparison between human and ranker annotation")
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)

args = parser.parse_args()

with open(
        f'results/Llama-3.2-3B-Instruct_{args.dataset}/loop_bf16/loop_manhattan_rank_bf16_0.jsonl',
        'r', encoding='utf-8') as f:
    ranking_list = [json.loads(line) for line in f if line.strip()]
with open(
        f'results/Llama-3.2-3B-Instruct_{args.dataset}/loop_bf16/loop_manhattan_contri_bf16_0.jsonl',
        'r', encoding='utf-8') as f:
    contri_list = [json.loads(line) for line in f if line.strip()]
with open(
        f'results/Llama-3.2-3B-Instruct_{args.dataset}/loop_bf16/loop_manhattan_path_bf16_0.jsonl',
        'r', encoding='utf-8') as f:
    path_list = [json.loads(line) for line in f if line.strip()]


with open(f'results/Llama-3.2-3B-Instruct_{args.dataset}/prediction_collection_bf16.json', 'r') as f:
    prediction_list = json.load(f)

with open(f'results/Llama-3.2-3B-Instruct_{args.dataset}/hem_answerable_collection_bf16.json', 'r') as f:
    hem_list = json.load(f)

shap_list = torch.load(f'results/Llama-3.2-3B-Instruct_{args.dataset}/shap_Qwen3-Reranker-8B_collection.pt')
full_stop_positions = torch.load(f'results/Llama-3.2-3B-Instruct_{args.dataset}/full_stop_positions.pt')
indices_to_remove = torch.load(f'results/Llama-3.2-3B-Instruct_{args.dataset}/indices_to_remove.pt')
manual_filtered = torch.load(f'results/Llama-3.2-3B-Instruct_{args.dataset}/manual_filtered.pt')

prediction_list = [v for i, v in enumerate(prediction_list) if i not in indices_to_remove]
hem_list = [v for i, v in enumerate(hem_list) if i not in indices_to_remove]
shap_list = [v for i, v in enumerate(shap_list) if i not in indices_to_remove]
full_stop_positions = [v for i, v in enumerate(full_stop_positions) if i not in indices_to_remove]
contri_list = [v for i, v in enumerate(contri_list) if i not in indices_to_remove]
path_list = [v for i, v in enumerate(path_list) if i not in indices_to_remove]
ranking_list = [v for i, v in enumerate(ranking_list) if i not in indices_to_remove]

obtained_num = 500
prediction_list = prediction_list[0: obtained_num]
hem_list = hem_list[0: obtained_num]
shap_list = shap_list[0: obtained_num]
full_stop_positions = full_stop_positions[0: obtained_num]
manual_filtered = manual_filtered[0:obtained_num]

ranking_list = [i['rankings'] for i in ranking_list[0:obtained_num]]
contri_list = [i['contri'] for i in contri_list[0:obtained_num]]
path_list = [i['path'] for i in path_list[0:obtained_num]]

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

all_subword_rank_phrase = []
all_subword_contri_phrase = []
all_subword_path_phrase = []
all_subword_tokens_phrase = []
all_piece_imp_phrase = []
all_piece_tokens_phrase = []

for s_idx, pred in enumerate(prediction_list):
    print(s_idx)
    context_tokens = np.array(pred['prompt_tokens'][start_list[s_idx]:end_list[s_idx]])

    rank_values = to_simplex_shift(rank_score_list[s_idx])
    contri_values = to_simplex_shift(contri_list[s_idx])
    path_values = to_simplex_shift(path_list[s_idx])

    subword_contri_phrase = []
    subword_token_phrase = []
    subword_rank_phrase = []
    subword_path_phrase = []
    piece_imp_phrase = []
    piece_tokens_phrase = []

    subword_pos = 0
    piece_pos = 0

    while piece_pos < len(shap_token_phrase[s_idx]):
        piece_buffer = shap_token_phrase[s_idx][piece_pos]
        piece_imp = shap_value_phrase[s_idx][piece_pos]
        subword_buffer = ""
        subword_contri = 0
        subword_path = 0
        subword_rank = []

        while subword_pos < len(context_tokens) and piece_pos < len(shap_token_phrase[s_idx]):
            subword_buffer += context_tokens[subword_pos]
            subword_contri += contri_values[subword_pos]
            subword_path += path_values[subword_pos]
            subword_rank.append(rank_values[subword_pos])
            subword_pos += 1

            # Keep extending piece_buffer until lengths are compatible
            while len(piece_buffer.strip()) < len(subword_buffer.strip()):
                piece_pos += 1
                piece_buffer += shap_token_phrase[s_idx][piece_pos]

            if subword_buffer.strip() == piece_buffer.strip():
                subword_contri_phrase.append(subword_contri)
                subword_path_phrase.append(subword_path)
                subword_rank_phrase.append(np.mean(subword_rank))
                subword_token_phrase.append(subword_buffer)
                piece_imp_phrase.append(piece_imp)
                piece_tokens_phrase.append(piece_buffer)
                piece_pos += 1
                break
    all_subword_rank_phrase.append(subword_rank_phrase)
    all_subword_contri_phrase.append(subword_contri_phrase)
    all_subword_path_phrase.append(subword_path_phrase)
    all_subword_tokens_phrase.append(subword_token_phrase)
    all_piece_imp_phrase.append(piece_imp_phrase)
    all_piece_tokens_phrase.append(piece_tokens_phrase)

all_subword_rank_word = []
all_subword_contri_word = []
all_subword_path_word = []
all_subword_tokens_word = []
all_piece_imp_word = []
all_piece_tokens_word = []

for s_idx, pred in enumerate(prediction_list):
    print(s_idx)
    context_tokens = np.array(pred['prompt_tokens'][start_list[s_idx]:end_list[s_idx]])

    rank_values = to_simplex_shift(rank_score_list[s_idx])
    contri_values = to_simplex_shift(contri_list[s_idx])
    path_values = to_simplex_shift(path_list[s_idx])

    subword_contri_word = []
    subword_token_word = []
    subword_rank_word = []
    subword_path_word = []
    piece_imp_word = []
    piece_tokens_word = []

    subword_pos = 0
    piece_pos = 0

    while piece_pos < len(shap_token_word[s_idx]):
        piece_buffer = shap_token_word[s_idx][piece_pos]
        piece_imp = shap_value_word[s_idx][piece_pos]
        subword_buffer = ""
        subword_contri = 0
        subword_path = 0
        subword_rank = []

        while subword_pos < len(context_tokens) and piece_pos < len(shap_token_word[s_idx]):
            subword_buffer += context_tokens[subword_pos]
            subword_contri += contri_values[subword_pos]
            subword_path += path_values[subword_pos]
            subword_rank.append(rank_values[subword_pos])
            subword_pos += 1

            # Keep extending piece_buffer until lengths are compatible
            while len(piece_buffer.strip()) < len(subword_buffer.strip()):
                piece_pos += 1
                piece_buffer += shap_token_word[s_idx][piece_pos]

            if subword_buffer.strip() == piece_buffer.strip():
                subword_contri_word.append(subword_contri)
                subword_path_word.append(subword_path)
                subword_rank_word.append(np.mean(subword_rank))
                subword_token_word.append(subword_buffer)
                piece_imp_word.append(piece_imp)
                piece_tokens_word.append(piece_buffer)
                piece_pos += 1
                break
    all_subword_rank_word.append(subword_rank_word)
    all_subword_contri_word.append(subword_contri_word)
    all_subword_path_word.append(subword_path_word)
    all_subword_tokens_word.append(subword_token_word)
    all_piece_imp_word.append(piece_imp_word)
    all_piece_tokens_word.append(piece_tokens_word)

p = 0.7

rank_rbo_phrase = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_rank_phrase[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_phrase[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    rank_rbo_phrase.append(rbo_val)

rank_rbo_initial = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_rank_initial[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_initial[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    rank_rbo_initial.append(rbo_val)

rank_rbo_word = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_rank_word[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_word[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    rank_rbo_word.append(rbo_val)

contri_rbo_phrase = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_contri_phrase[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_phrase[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    contri_rbo_phrase.append(rbo_val)

contri_rbo_initial = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_contri_initial[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_initial[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    contri_rbo_initial.append(rbo_val)

contri_rbo_word = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_contri_word[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_word[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    contri_rbo_word.append(rbo_val)

path_rbo_phrase = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_path_phrase[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_phrase[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    path_rbo_phrase.append(rbo_val)

path_rbo_initial = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_path_initial[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_initial[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    path_rbo_initial.append(rbo_val)

path_rbo_word = []
for s_idx in range(obtained_num):
    rank1 = list(np.argsort(all_subword_path_word[s_idx])[::-1])
    rank2 = list(np.argsort(all_piece_imp_word[s_idx])[::-1])
    rbo_val = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
    path_rbo_word.append(rbo_val)

labels = [1 if s > 0.5 else 0 for s in hem_list]

X = np.column_stack([
    rank_rbo_initial,rank_rbo_word,rank_rbo_phrase,
    contri_rbo_initial,contri_rbo_word,contri_rbo_phrase,
    path_rbo_initial,path_rbo_word,path_rbo_phrase,
])

X_filtered = np.array([X[i,:] for i in range(len(X)) if manual_filtered[i]==1])
labels_filtered = [labels[i] for i in range(len(labels)) if manual_filtered[i]==1]

for col in range(X.shape[1]):
    print(col)
    auprc = average_precision_score(labels, X[:, col])
    auprc_filtered = average_precision_score(labels_filtered, X_filtered[:, col])
    print('original auprc:', auprc, ' filtered auprc:', auprc_filtered)
    auroc = roc_auc_score(labels, X[:, col])
    auroc_filtered = roc_auc_score(labels_filtered, X_filtered[:, col])
    print('original auroc:', auroc, ' filtered auprc', auroc_filtered)

print(len(labels_filtered)/len(labels))