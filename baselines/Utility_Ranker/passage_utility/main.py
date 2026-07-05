from pathlib import Path
import torch
import sys
import json
import os

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utility_Ranker.passage_utility.dataset_collection import PosNegDataset
from Utility_Ranker.passage_utility.load_data import load_ragqa
from Utility_Ranker.passage_utility.models.bert_ranker import BertRanker
from Utility_Ranker.passage_utility.reward_learner.vallina_bert import VanillaBert
from Utility_Ranker.passage_utility.swag import utils

import numpy as np
import argparse
from tqdm import trange
import warnings
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch

# 定义 collate_fn 函数来填充序列，使它们的长度一致
def collate_fn(batch):
    input_ids1 = [item['input_ids1'] for item in batch]
    input_ids2 = [item['input_ids2'] for item in batch]
    attention_mask1 = [item['attention_mask1'] for item in batch]
    attention_mask2 = [item['attention_mask2'] for item in batch]
    token_type_ids1 = [item['token_type_ids1'] for item in batch]
    token_type_ids2 = [item['token_type_ids2'] for item in batch]
    score1 = [item['score1'] for item in batch]
    score2 = [item['score2'] for item in batch]
    acc1 = [item['acc1'] for item in batch]
    acc2 = [item['acc2'] for item in batch]
    targets = [item['targets'] for item in batch]

    # 使用 pad_sequence 来确保 input_ids 和 attention_mask 都填充到相同长度
    input_ids1 = pad_sequence(input_ids1, batch_first=True, padding_value=0)
    input_ids2 = pad_sequence(input_ids2, batch_first=True, padding_value=0)
    attention_mask1 = pad_sequence(attention_mask1, batch_first=True, padding_value=0)
    attention_mask2 = pad_sequence(attention_mask2, batch_first=True, padding_value=0)
    token_type_ids1 = pad_sequence(token_type_ids1, batch_first=True, padding_value=0)
    token_type_ids2 = pad_sequence(token_type_ids2, batch_first=True, padding_value=0)

    # 将 labels 合并成一个 tensor
    return {
        'input_ids1': input_ids1,
        'input_ids2': input_ids2,
        'attention_mask1': attention_mask1,
        'attention_mask2': attention_mask2,
        'token_type_ids1': token_type_ids1,
        'token_type_ids2': token_type_ids2,
        'targets': torch.stack(targets),
        'score1': torch.stack(score1),
        'score2': torch.stack(score2),
        'acc1': torch.stack(acc1),
        'acc2': torch.stack(acc2),
    }

def preprocess_data(questions, answers, tokenizer, max_length=512):
    # 将 question 和 answer 组合成 BERT 输入格式（例如 [CLS] question [SEP] answer [SEP]）
    inputs = [q + " " + a for q, a in zip(questions, answers)]
    encodings = tokenizer(
        inputs,
        padding='max_length',  # 填充到最大长度
        truncation=True,       # 截断长序列
        max_length=max_length, # 最大长度（根据你的模型和硬件调整）
        return_tensors='pt'    # 返回 PyTorch 张量
    )
    return encodings


def resolve_split_path(path_template, split_name):
    normalized = split_name
    if split_name in {"dev", "valid", "validation"}:
        normalized = "val"
    return path_template.replace("SPLIT", normalized) if "SPLIT" in path_template else path_template


def resolve_checkpoint_path(args):
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.suffix != ".pt":
            checkpoint_path = checkpoint_path.with_suffix(".pt")
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(args.save_dir) / checkpoint_path.name
    else:
        checkpoint_path = Path(args.save_dir) / f"{args.model_name}best-val-acc-model.pt"
    return checkpoint_path


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default="",
    )
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument(
        '--save_dir',
        type=str,
        default="",
    )
    parser.add_argument('--checkpoint', type=str, default= None)
    parser.add_argument('--num_shards', type=int, default=0)          

    parser.add_argument('--lr_init', type=float,
                        default=5e-5, help='initial learning rate')
    parser.add_argument('--ilr', type=float, default=1e-4,
                        metavar='N', help='learning rate for interaction')
    parser.add_argument('--wd', type=float, default=1e-2,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=3,
                        metavar='N', help='SWA start epoch number')
                  
    parser.add_argument('--proportion', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to train')
    parser.add_argument('--proportion_dev', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to from dev set')                        

    parser.add_argument('--batch_size', type=int, default=300,
                        metavar='N', help='input batch size')
    parser.add_argument('--pretrained_model', type=str, default=str(PROJECT_ROOT / 'passage_utility' / 'bert-base-uncased'),
                        metavar='N', help='the name of pretrained_model')
    parser.add_argument('--mode', type=str, default='test',
                        metavar='N', help='dataset')
    parser.add_argument('--test_mode', type=str, default='test',
                        metavar='N', help='dataset')

    parser.add_argument('--sample_nums', type=int, default=20, metavar='N')
    parser.add_argument('--model_name', type=str, default='vanilla_bert')
    parser.add_argument('--margin', type=float, default=0.1, metavar='N')  
    parser.add_argument('--stop_epochs', type=int, default=2, metavar='N')

    parser.add_argument('--interactive', type=str2bool, default=False, metavar='N')
    parser.add_argument('--do_train', type=str2bool, default=False, metavar='N')
    parser.add_argument('--do_test', type=str2bool, default=True, metavar='N')
    parser.add_argument('--pool_sample', type=str2bool, default=False, metavar='N')    

    parser.add_argument('--format_ques', type=str2bool, default=False, metavar='N')    
    parser.add_argument('--add_title', type=str2bool, default=False, metavar='N')
    parser.add_argument('--top_n', type=int, default=5,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--reference_rank', type=str, default='nli',
                        help="criteria to use as reference utility (other possible values: rl-nli, rl, nli, acc-nli, acc_LM-nli, acc, acc-ties).")            
    parser.add_argument('--output_pred_utilities', type=str2bool, default=False, metavar='N')
    parser.add_argument('--combine_loss', type=str, default=None,
                        help="whether to combine ranking loss with BE/MSE(values: be, mse).")      
    parser.add_argument('--weight_rank', type=float, default=1,
                        help="weight of ranking loss.")   
    parser.add_argument('--weight_aux', type=float, default=0,
                        help="weight of combined auxiliary loss.")   
    parser.add_argument(
        "--model_select", type=str, default="combined",
        choices=['combined', 'error', 'rank'],
        help="Model selection criteria for improvement evaluation.")                                                
    parser.add_argument(
        '--utility_output',
        type=str,
        default=None,
        help='Optional output path for predicted utilities during test mode.',
    )
    parser.add_argument(
        '--score_output',
        type=str,
        default=None,
        help='Optional output path for raw ranker scores during test mode.',
    )

    args = parser.parse_args()

    print(args)
    directory_name = os.path.join(args.save_dir,'logs')
    log_mode = 'TRAIN' if args.do_train else f'EVAL:{args.test_mode}'
    log_file_name = os.path.join(directory_name, f'{args.model_name}-{log_mode}.log')
    os.makedirs(directory_name, exist_ok=True)
    log_file = open(log_file_name, 'w')
    log_file.write(f'{args}')
    print(f'Logging run config into: {log_file_name}\n')

    warnings.filterwarnings('ignore')
    if args.interactive:
        print('Not implemented')
        exit(0)
        qa_list, ref_values = load_ragqa(
            resolve_split_path(args.input_file, args.mode),
            args.top_n,
            args.reference_rank,
            interactive=True,
        )
            
    if args.do_train:
        train_input_path = resolve_split_path(args.input_file, args.mode)
        dev_input_path = resolve_split_path(args.input_file, 'val')
        data_pair = load_ragqa(
            train_input_path, args.top_n, args.reference_rank, interactive=False,
                    shards=args.num_shards, single_net=(args.weight_rank==0), add_title=args.add_title)
        dev_pair = load_ragqa(
            dev_input_path, args.top_n, args.reference_rank, interactive=False,
                    single_net=(args.weight_rank==0), add_title=args.add_title)
        
        print('training data size', len(data_pair))
        print('dev data size', len(dev_pair))
        data_loader = DataLoader(PosNegDataset(
            data_pair[:int(len(data_pair)*args.proportion)], args.pretrained_model), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(PosNegDataset(
            dev_pair[:int(len(dev_pair)*args.proportion_dev)], args.pretrained_model), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    base_model = BertRanker(args.pretrained_model)
    base_model.to(device)

    if args.do_train:
        print('start training')
        if 'vanilla_bert' in args.model_name.lower():
            model = VanillaBert(base_model, lr=args.lr_init, ilr=args.ilr, epochs=args.epochs,
                             pretrained_model=args.pretrained_model, device=device, weight_decay=args.wd, 
                             margin=args.margin, combine_loss=args.combine_loss, weight_rank=args.weight_rank, 
                             weight_aux=args.weight_aux, model_select=args.model_select)
            model.to(device)
        else:
            print('Unimplemented')
            exit(0)
            
        checkpoint_path = resolve_checkpoint_path(args) if args.checkpoint else None
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = None
        model.train(data_loader, valid_loader=valid_loader, save_dir=args.save_dir,
                stop_epochs=args.stop_epochs, save_name=args.model_name, checkpoint =checkpoint,
                log_file=log_file)

    if args.do_test:
        data =[]
        data_path = resolve_split_path(args.input_file, args.test_mode)
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        print('start testing...')

        checkpoint_path = resolve_checkpoint_path(args)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Set --checkpoint explicitly or train first."
            )
        checkpoint = torch.load(checkpoint_path)
            
        if 'vanilla_bert' in args.model_name.lower():
            print('vallina bert!!')
            model = VanillaBert(base_model, lr=args.lr_init, ilr = args.ilr, epochs=args.epochs,
                             device=device, pretrained_model=args.pretrained_model, weight_decay=args.wd, 
                             margin=args.margin, combine_loss=args.combine_loss, weight_rank=args.weight_rank, weight_aux=args.weight_aux)
        else:
            print('Unimplemented')
            exit(0)

        model.load_state_dict(checkpoint['state_dict'])
        print('finish loading model!')
        tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
        res, acc = 0, 0
        whole_answers, question, cumcnt = [], [], []
        prev = 0
        for question_id in trange(len(data)):
            entry = data[question_id]
            if args.add_title:
                pooled = [
                    (ctx.get("title", "") + "\n" + ctx["text"]).strip()
                    for ctx in entry["ctxs"][:args.top_n]
                ]
            else:
                pooled = [ctx["text"] for ctx in entry["ctxs"][:args.top_n]]
                if not pooled and "context" in entry:
                    pooled = [entry["context"]]
            if args.format_ques:
                ques = entry["question"]
                ques = ques[0].upper() + ques[1:] + '?'
                question.extend([ques]*len(pooled))
            else:
                question.extend([entry["question"]]*len(pooled))
            whole_answers.extend(pooled)
            prev += len(pooled)
            cumcnt.append(prev)
        print('total nums', len(whole_answers))
        assert len(question) == len(whole_answers)
        # encodings = preprocess_data(question, whole_answers, tokenizer)
        utilities, all_scores = model.get_utilities(
            test_data=whole_answers, question=question, sample_nums=args.sample_nums)
        utilities_list = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in utilities]
        print(len(utilities_list))
        all_scores = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in all_scores]
        utility_output = args.utility_output or os.path.join(
            args.save_dir, f'{args.model_name}_{args.test_mode}_utilities.json'
        )
        score_output = args.score_output or os.path.join(
            args.save_dir, f'{args.model_name}_{args.test_mode}_scores.json'
        )
        os.makedirs(Path(utility_output).parent, exist_ok=True)
        os.makedirs(Path(score_output).parent, exist_ok=True)
        with open(utility_output, 'w', encoding='utf-8') as f:
            json.dump({"utility_score": utilities_list}, f, ensure_ascii=False, indent=4)
        with open(score_output, 'w', encoding='utf-8') as f:
            json.dump({"raw_scores": all_scores}, f, ensure_ascii=False, indent=4)

    log_file.close()
