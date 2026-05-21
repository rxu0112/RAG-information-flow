import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from utils import *
from rbo import rbo

# Instruction "answer the question no more than x words" 
# combine Q and A, Qwen2.5 7B, vectara/hallucination_evaluation_model

torch._dynamo.config.cache_size_limit = 99999

def find_section(tokens, start, keyword):
    matches = [i + start for i, t in enumerate(tokens[start:]) if t.strip() == keyword]
    if len(matches) == 0:
        return None
    elif len(matches) > 1:
        return None
    return matches[0]

def generate_answer(question, context):
    prompt = f"Answer the question in no more than five words. Context: {context} Question: {question} Answer:"
    encoding = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)

    prompt_token_ids = input_ids[0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    outputs = model.generate(
        input_ids,
        do_sample=False,  # Greedy decoding
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=10,
    )

    generated_ids = outputs[0][input_ids.shape[1]:]
    pred_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    first_sentence = pred_answer.split('.')[0].strip()

    return {'pred_answer': pred_answer, 
            'prompt_tokens': tokens, 
            'generated_tokens': [tokenizer.decode([tid]) for tid in generated_ids.tolist()],
            'first_sentence': first_sentence}

def generate_statement(question, answer):
    statement_prompt = (
        f"Convert the following Q&A into a single factual sentence.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Statement:"
    )

    encoding = merge_tokenizer(statement_prompt, return_tensors="pt").to(device)
    outputs = merge_model.generate(
        **encoding,
        do_sample=False,
        max_new_tokens=128,
        pad_token_id=merge_tokenizer.pad_token_id,
        eos_token_id=merge_tokenizer.eos_token_id,
    )
    statement = merge_tokenizer.decode(outputs[0][encoding["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return statement


parser = argparse.ArgumentParser(description="Inference")
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct','Llama-3-8B-Instruct','gemma-3-4B-it'], required=True)
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', '{args.dataset}'], required=True)
args = parser.parse_args()

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
save_path = f'results/{args.model}_{args.dataset}/prediction_faithfulness_bf16.json'

device_num = 8

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

with open(f'processed_data/{args.dataset}/{args.dataset}_prepared.json') as f:
    dataset_list = json.load(f)

with open(f'results/{args.model}_{args.dataset}/hem_answerable_collection_bf16.json', 'r') as f:
    hem_list = json.load(f)

shap_list = torch.load(f'results/{args.model}_{args.dataset}/shap_Qwen3-Reranker-8B_collection.pt', weights_only=False)
full_stop_positions = torch.load(f'results/{args.model}_{args.dataset}/full_stop_positions.pt')

obtained_num = min([len(ranking_list), len(contri_list), len(path_list), len(prediction_list)])
dataset_list = dataset_list[0: obtained_num]
prediction_list = prediction_list[0: obtained_num]
hem_list = hem_list[0: obtained_num]
shap_list = shap_list[0: obtained_num]
full_stop_positions = full_stop_positions[0: obtained_num]
ranking_list = [i['rankings'] for i in ranking_list[0:obtained_num]]
contri_list = [i['contri'] for i in contri_list[0:obtained_num]]
path_list = [i['path'] for i in path_list[0:obtained_num]]

indices_to_remove = torch.load(f'results/{args.model}_{args.dataset}/indices_to_remove.pt')
dataset_list = [v for i, v in enumerate(dataset_list) if i not in indices_to_remove]
prediction_list = [v for i, v in enumerate(prediction_list) if i not in indices_to_remove]
hem_list = [v for i, v in enumerate(hem_list) if i not in indices_to_remove]
shap_list = [v for i, v in enumerate(shap_list) if i not in indices_to_remove]
full_stop_positions = [v for i, v in enumerate(full_stop_positions) if i not in indices_to_remove]
contri_list = [v for i, v in enumerate(contri_list) if i not in indices_to_remove]
path_list = [v for i, v in enumerate(path_list) if i not in indices_to_remove]
ranking_list = [v for i, v in enumerate(ranking_list) if i not in indices_to_remove]

top_hem_indices = sorted(range(len(hem_list)), key=lambda i: hem_list[i], reverse=True)[:100]
hem_list = [hem_list[i] for i in top_hem_indices]
shap_list = [shap_list[i] for i in top_hem_indices]
dataset_list = [dataset_list[i] for i in top_hem_indices]
prediction_list = [prediction_list[i] for i in top_hem_indices]
full_stop_positions = [full_stop_positions[i] for i in top_hem_indices]
contri_list = [contri_list[i] for i in top_hem_indices]
path_list = [path_list[i] for i in top_hem_indices]
ranking_list = [ranking_list[i] for i in top_hem_indices]

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

rank_score_max_list = []
for i, rankings in enumerate(ranking_list):
    score_list = []
    for t in rankings:
        score = np.empty_like(t).copy()
        score[t] = np.arange(len(t), 0, -1)
        score_list.append(to_simplex_shift(score[start_list[i]:end_list[i]]))
    arr = np.array(score_list, dtype=object)
    max_vals = arr.max(axis=0)
    rank_score_max_list.append(max_vals)
rank_score_list = rank_score_max_list

contri_max_list = []
for i, contri in enumerate(contri_list):
    score_list = []
    for score in contri:
        score_list.append(to_simplex_shift(score[start_list[i]:end_list[i]]))
    arr = np.array(score_list, dtype=object)
    max_vals = arr.max(axis=0)
    contri_max_list.append(max_vals)
contri_list = contri_max_list


path_max_list = []
for i, path in enumerate(path_list):
    score_list = []
    for score in path:
        score_list.append(to_simplex_shift(score[start_list[i]:end_list[i]]))
    arr = np.array(score_list, dtype=object)
    max_vals = arr.max(axis=0)
    path_max_list.append(max_vals)
path_list = path_max_list

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

# load models
model_path = 'MODEL_PATH_PLACEHOLDER' # path of args.model
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
model.eval()
model.to(device)

# Load merge model and tokenizer
merge_model_path = 'MODEL_PATH_PLACEHOLDER' # path of Qwen2.5-7B
merge_model = AutoModelForCausalLM.from_pretrained(merge_model_path, torch_dtype=torch.float16)
merge_tokenizer = AutoTokenizer.from_pretrained(merge_model_path)
if merge_tokenizer.pad_token is None:
    merge_tokenizer.pad_token = merge_tokenizer.eos_token
    merge_model.resize_token_embeddings(len(merge_tokenizer))
merge_model.eval()
merge_model.to(device)

# Prediction and save
prediction = []
print("Predicting\n")

for s_idx, example in tqdm(enumerate(dataset_list)):
    question = example["question"]
    context_tokens = all_subword_tokens_initial[s_idx]

    rank_vals = all_subword_rank_initial[s_idx]
    contri_vals = all_subword_contri_initial[s_idx]
    path_vals = all_subword_path_initial[s_idx]

    top_rank_indices = sorted(range(len(rank_vals)), key=lambda i: rank_vals[i], reverse=True)[:3]
    low_rank_indices = sorted(range(len(rank_vals)), key=lambda i: rank_vals[i])[:3]
    top_contri_indices = sorted(range(len(contri_vals)), key=lambda i: contri_vals[i], reverse=True)[:3]
    low_contri_indices = sorted(range(len(contri_vals)), key=lambda i: contri_vals[i])[:3]
    top_path_indices = sorted(range(len(path_vals)), key=lambda i: path_vals[i], reverse=True)[:3]
    low_path_indices = sorted(range(len(path_vals)), key=lambda i: path_vals[i])[:3]

    context_tokens_wo_top_rank = [context_tokens[i] for i in range(len(context_tokens)) if i not in top_rank_indices]
    context_tokens_wo_low_rank = [context_tokens[i] for i in range(len(context_tokens)) if i not in low_rank_indices]
    context_wo_top_rank = ''.join(context_tokens_wo_top_rank)
    context_wo_low_rank = ''.join(context_tokens_wo_low_rank)

    context_tokens_wo_top_contri = [context_tokens[i] for i in range(len(context_tokens)) if i not in top_contri_indices]
    context_tokens_wo_low_contri = [context_tokens[i] for i in range(len(context_tokens)) if i not in low_contri_indices]
    context_wo_top_contri = ''.join(context_tokens_wo_top_contri)
    context_wo_low_contri = ''.join(context_tokens_wo_low_contri)

    context_tokens_wo_top_path = [context_tokens[i] for i in range(len(context_tokens)) if i not in top_path_indices]
    context_tokens_wo_low_path = [context_tokens[i] for i in range(len(context_tokens)) if i not in low_path_indices]
    context_wo_top_path = ''.join(context_tokens_wo_top_path)
    context_wo_low_path = ''.join(context_tokens_wo_low_path)

    is_impossible = example["is_impossible"]
    true_answers = [ans["text"].strip() for ans in example["answers"]]
    answer_info_wo_top_rank = generate_answer(question, context_wo_top_rank)
    answer_info_wo_low_rank = generate_answer(question, context_wo_low_rank)
    answer_info_wo_top_contri = generate_answer(question, context_wo_top_contri)
    answer_info_wo_low_contri = generate_answer(question, context_wo_low_contri)
    answer_info_wo_top_path = generate_answer(question, context_wo_top_path)
    answer_info_wo_low_path = generate_answer(question, context_wo_low_path)
    
    merged_true_answers = []
    for true_ans in true_answers:
        merged_true_answers.append(generate_statement(question, true_ans))
    merged_prediction_wo_top_rank = generate_statement(question, answer_info_wo_top_rank["first_sentence"])
    merged_prediction_wo_low_rank = generate_statement(question, answer_info_wo_low_rank["first_sentence"])
    merged_prediction_wo_top_contri = generate_statement(question, answer_info_wo_top_contri["first_sentence"])
    merged_prediction_wo_low_contri = generate_statement(question, answer_info_wo_low_contri["first_sentence"])
    merged_prediction_wo_top_path = generate_statement(question, answer_info_wo_top_path["first_sentence"])
    merged_prediction_wo_low_path = generate_statement(question, answer_info_wo_low_path["first_sentence"])

    sample_data = {
        'is_impossible': is_impossible,
        'merged_true_answers': merged_true_answers,
        'merged_prediction_wo_top_rank': merged_prediction_wo_top_rank,
        'merged_prediction_wo_low_rank': merged_prediction_wo_low_rank,
        'merged_prediction_wo_top_contri': merged_prediction_wo_top_contri,
        'merged_prediction_wo_low_contri': merged_prediction_wo_low_contri,
        'merged_prediction_wo_top_path': merged_prediction_wo_top_path,
        'merged_prediction_wo_low_path': merged_prediction_wo_low_path,
    }
    prediction.append(sample_data)

with open(save_path, 'w') as f:
    json.dump(prediction, f, indent=2)
