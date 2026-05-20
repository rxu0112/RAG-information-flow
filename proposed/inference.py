import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json

# Instruction "answer the question no more than x words" 
# combine Q and A, Qwen2.5 7B, vectara/hallucination_evaluation_model

torch._dynamo.config.cache_size_limit = 99999

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
parser.add_argument('--model', type=str, choices=['Llama-3.2-3B-Instruct','gemma-3-4B-it','Llama-3-8B-Instruct'], required=True)
parser.add_argument('--dataset', type=str, choices=['squad2', 'hotpot', 'msmarco'], required=True)

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu' # specificy your device

# load data
save_path = f'results/{args.model}_{args.data}/prediction_collection_bf16.json'
data_path = f"processed_data/{args.data}_prepared.json"
dataset = load_dataset("json", data_files={"validation": data_path})

if args.data == "squad2":
    dataset = dataset["validation"].select(range(0, 50000))
elif args.data == "msmarco":
    dataset = dataset["validation"].select(range(0, 45000))
elif args.data == "hotpot":
    dataset = dataset["validation"].select(range(0, 40000))

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

for example in tqdm(dataset, total=len(dataset)):
    question = example["question"]
    context = example["context"]
    is_impossible = example["is_impossible"]
    true_answers = [ans["text"].strip() for ans in example["answers"]]
    answer_info = generate_answer(question, context)

    merged_true_answers = []
    for true_ans in true_answers:
        merged_true_answers.append(generate_statement(question, true_ans))
    merged_prediction = generate_statement(question, answer_info["first_sentence"])

    sample_data = {
        'is_impossible': is_impossible,
        'true_answers': true_answers,
        'predicted_answer': answer_info["pred_answer"],
        'question': question,
        'context': context,
        'prompt_tokens': answer_info["prompt_tokens"],
        'merged_true_answers': merged_true_answers,
        'merged_prediction': merged_prediction,
        'generated_tokens': answer_info["generated_tokens"],
        'first_sentence': answer_info["first_sentence"]
    }
    prediction.append(sample_data)

with open(save_path, 'w') as f:
    json.dump(prediction, f, indent=2)


