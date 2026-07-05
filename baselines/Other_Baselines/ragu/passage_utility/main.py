from importlib import util
from scipy.stats import mode
import torch
from Other_Baselines.ragu.passage_utility.load_data import load_ragqa
import sys
import json
sys.path.append('../utils')
from Other_Baselines.ragu.utils.utils import save_file_jsonl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"  # 指定使用第1张GPU
import numpy as np
import argparse
from Other_Baselines.ragu.passage_utility.reward_learner.vallina_bert import VanillaBert
from Other_Baselines.ragu.passage_utility.models.bert_ranker import BertRanker
from tqdm import trange
import warnings
import Other_Baselines.ragu.passage_utility.swag.utils as utils
# from evaluator.evaluation import evaluateReward
from torch.utils.data import DataLoader
from Other_Baselines.ragu.passage_utility.dataset_collection import PosNegDataset
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
    inputs = [q + " " + a for q, a in zip(questions, answers)]
    encodings = tokenizer(
        inputs,
        padding='max_length', 
        truncation=True,       
        max_length=max_length, 
        return_tensors='pt'    
    )
    return encodings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='...')
    parser.add_argument('--cache_dir', type=str, default='./cache/')
    parser.add_argument('--save_dir', type=str, default='ragu/new_data/hotpot/checkpoint')
    parser.add_argument('--checkpoint', type=str, default= None)
    parser.add_argument('--num_shards', type=int, default=0)          

    parser.add_argument('--lr_init', type=float,
                        default=5e-5, help='initial learning rate')
    parser.add_argument('--ilr', type=float, default=1e-4,
                        metavar='N', help='learning rate for interaction')
    parser.add_argument('--wd', type=float, default=1e-2,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=2,
                        metavar='N', help='SWA start epoch number')
                  
    parser.add_argument('--proportion', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to train')
    parser.add_argument('--proportion_dev', type=float, default=1.0,
                        metavar='N', help='the proportion of data used to from dev set')                        

    parser.add_argument('--batch_size', type=int, default=300,
                        metavar='N', help='input batch size')
    parser.add_argument('--pretrained_model', type=str, default='ragu/passage_utility/bert-base-uncased',
                        metavar='N', help='the name of pretrained_model')
    parser.add_argument('--mode', type=str, default='train',
                        metavar='N', help='dataset')
    parser.add_argument('--test_mode', type=str, default='test',
                        metavar='N', help='dataset')

    parser.add_argument('--sample_nums', type=int, default=20, metavar='N')
    parser.add_argument('--model_name', type=str, default='vanilla_bert')
    parser.add_argument('--margin', type=float, default=0.1, metavar='N')  
    parser.add_argument('--stop_epochs', type=int, default=2, metavar='N')

    parser.add_argument('--interactive', type=bool, default=False, metavar='N')
    parser.add_argument('--do_train', type=bool, default=True, metavar='N')
    parser.add_argument('--do_test', type=bool, default=False, metavar='N')
    parser.add_argument('--pool_sample', type=bool, default=False, metavar='N')    

    parser.add_argument('--format_ques', type=bool, default=False, metavar='N')    
    parser.add_argument('--add_title', type=bool, default=False, metavar='N')
    parser.add_argument('--top_n', type=int, default=5,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--reference_rank', type=str, default='nli',
                        help="criteria to use as reference utility (other possible values: rl-nli, rl, nli, acc-nli, acc_LM-nli, acc, acc-ties).")            
    parser.add_argument('--output_pred_utilities', type=bool, default=False, metavar='N')
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

    args = parser.parse_args()

    print(args)
    directory_name = os.path.join(args.save_dir,'logs')
    log_mode = 'TRAIN' if args.do_train else f'EVAL:{args.test_mode}'
    log_file_name = os.path.join(directory_name, f'{args.model_name}-{log_mode}.log')
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
            print(f"Directory '{directory_name}' created successfully.")
        except FileExistsError:
            print(f"Directory '{directory_name}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{directory_name}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
    log_file = open(log_file_name, 'w')
    log_file.write(f'{args}')
    print(f'Logging run config into: {log_file_name}\n')

    warnings.filterwarnings('ignore')
    if args.interactive:
        print('Not implemented')
        exit(0)
        # load data
        qa_list, ref_values = load_ragqa(
            args.input_file.replace('SPLIT', args.mode), args.top_n, args.reference_rank, interactive=True)
            
    # data
    if args.do_train:
        data_pair = load_ragqa(
            args.input_file.replace('SPLIT', args.mode), args.top_n, args.reference_rank, interactive=False, 
                    shards=args.num_shards, single_net=(args.weight_rank==0), add_title=args.add_title)
        dev_pair = load_ragqa(
            args.input_file.replace('SPLIT', 'dev'), args.top_n, args.reference_rank, interactive=False,
                    single_net=(args.weight_rank==0), add_title=args.add_title)
        
        print('training data size', len(data_pair))
        print('dev data size', len(dev_pair))
        data_loader = DataLoader(PosNegDataset(
            data_pair[:int(len(data_pair)*args.proportion)], args.pretrained_model), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(PosNegDataset(
            dev_pair[:int(len(dev_pair)*args.proportion_dev)], args.pretrained_model), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # select data(batch or single) according to query strategy, initial data 0.1%?
    # querier
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    # device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    base_model = BertRanker(args.pretrained_model)
    base_model.to(device)

    # train
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
            
        if args.checkpoint:
            checkpoint = torch.load(os.path.join(
                args.save_dir, args.checkpoint+'.pt'))
        else:
            checkpoint = None
        model.train(data_loader, valid_loader=valid_loader, save_dir=args.save_dir,
                stop_epochs=args.stop_epochs, save_name=args.model_name, checkpoint =checkpoint,
                log_file=log_file)


    if args.do_test:
        data =[]
        data_path = args.input_file
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        print('start testing...')

        checkpoint = torch.load('ragu/new_data/msmarco/checkpoint3/vanilla_bertbest-val-acc-model.pt')
            
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
                pooled = [ctx["title"] + "\n" + ctx["text"] for ctx in entry["ctxs"][:args.top_n]]
            else:
                pooled = [entry['context']]
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
        with open('ragu/new_data/msmarco/llamma_bert_test_result.json', 'w', encoding='utf-8') as f:
            json.dump(utilities_list, f, ensure_ascii=False, indent=4)
        with open('ragu/new_data/msmarco/llamma_bert_test_result_scores.json', 'w', encoding='utf-8') as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=4)

    log_file.close()
