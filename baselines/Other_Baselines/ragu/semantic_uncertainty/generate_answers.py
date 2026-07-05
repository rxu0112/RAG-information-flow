"""Sample answers from LLMs on QA task."""
import gc
import os
import logging
import random
from tqdm import tqdm

import numpy as np
import torch
import wandb

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import multiprocessing

from Other_Baselines.ragu.semantic_uncertainty.uncertainty.data.data_utils import load_ds, load_ds_precomputed
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.utils import utils
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures import p_true as p_true_utils
from Other_Baselines.ragu.semantic_uncertainty.compute_uncertainty_measures import main as main_compute
import sys
sys.path.append('../utils')
from Other_Baselines.ragu.utils import PROMPT_DICT, save_file_jsonl, load_file

utils.setup_logger()

def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0

def main(args):

    # Setup run.
    experiment_details = {'args': args}
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(args.random_seed)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(args.random_seed)
    #Fix torch random seed
    torch.manual_seed(args.random_seed)
    #user = os.environ['USER']
    pod_jobid = os.getenv('POD_NAME')
    scratch_dir = '/mnt/data/obqa/outputs'
    if not os.path.exists(f"{scratch_dir}/uncertainty"):
        os.makedirs(f"{scratch_dir}/uncertainty")

    wandb.init(
        entity=args.entity,
        project="semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug",
        dir=f"{scratch_dir}/uncertainty",
        config=args,
        notes=f'pod_id: {pod_jobid}, experiment_lot: {args.experiment_lot}',
    )
    logging.info('Finished wandb init.')
    print('Wandb entity', args.entity)
    print('Wandb run_id', wandb.run.id)
    print('Wandb run_dir', wandb.run.dir)


    # Get accuracy metric.
    metric = utils.get_metric(args.metric)

    # Load dataset.

    fewshots_ids = []
    fewshot_prompt_rag = None
    if args.fewshots:
        fewshots = load_file(args.fewshots)
        fewshots_ids = [x['q_id'] for x in fewshots]
        fewshot_prompt_rag = "".join(['{}\nAnswer: {}\n\n'.format(x['question'],x['answers'][0]) for x in fewshots])

    if not args.precomputed_gen:
        train_dataset, validation_dataset = load_ds(
            args.dataset, add_options=args.use_mc_options, seed=args.random_seed)
    else:
        train_dataset, validation_dataset = load_ds_precomputed(args.dataset,
                                                args.eval_mode,
                                                args.proportion,
                                                args.most_likely_file, 
                                                args.samples_file, 
                                                args.original_file,
                                                args.top_n, 
                                                args.utilities_file if args.compute_utility else None,
                                                args.ood_train_dataset)
        if fewshots_ids:
            train_dataset = [x for x in train_dataset if not x['q_id'] in fewshots_ids]
            print('Fewshots removed, ', len(train_dataset))


    # Get indices of answerable and unanswerable questions and construct prompt.
    if args.answerable_only:
        unanswerable_indices = []
        val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
        del val_unanswerable
        validation_dataset = [validation_dataset[i] for i in val_answerable]

    # Initialize model.
    model = utils.init_model(args)

    # Initialize prompt for p_true baseline.
    if args.compute_p_true:
        logging.info(80*'#')
        logging.info('Constructing few-shot prompt for p_true.')
        # We use already pre-computed sample in the given train file.
        p_true_few_shot_prompt, p_true_responses, len_p_true = p_true_utils.construct_few_shot_prompt_RAG_given_list(
                        model, train_dataset, args.num_generations, args.acc_LM)        

        experiment_details['p_true_responses'] = p_true_responses
        experiment_details['p_true_few_shot_prompt'] = p_true_few_shot_prompt
        logging.info('Finished constructing few-shot prompt for p_true.')
        logging.info(80*'#')
        logging.info('p_true_few_shot_prompt: %s', p_true_few_shot_prompt)
        logging.info(80*'#')

    # Start answer generation.
    logging.info(80 * '=')
    logging.info('Generating answers: ')
    logging.info(80 * '=')
    for dataset_split in ['train', 'validation']:
        logging.info(80 * 'x')
        logging.info('Starting with dataset_split %s.', dataset_split)
        logging.info(80 * 'x')

        # This will store all input data and model predictions.
        accuracies, generations, results_dict, p_trues  = [], {}, {}, []
        utilitites_pred = {
            'nli_utility_predicted': [],
            'acc-nli_utility_predicted': [],
            'acc_LM_utility_predicted': [],           
            'acc_LM-nli_utility_predicted': [],
            'utility_confidence_predicted': [],
            'acc_utility_predicted': [],
            'mean_utility_predicted': [],
            'score_utility_predicted': [],
            'ragqa_ppl_utility': [],
            'ragqa_MSP_utility': [],
            'ragqa_PMI_utility': [],
            'ragqa_RenyiNeg_utility': [],
            'ragqa_FisherRao_utility': [],
        } 

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                logging.info('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = list(set(remaining_answerable) | set(unanswerable_indices))

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        # Evaluate over random subset of the datasets.
        indices = random.sample(possible_indices, min(args.num_samples, len(dataset)))
        experiment_details[dataset_split] = {'indices': indices}

        if args.num_samples > len(dataset):
            logging.warning('Not enough samples in dataset. Using all %d samples.', len(dataset))

        # get a sample to train the confidence predictor
        conf_model = None
        if args.train_confidence:
            new_possible_indices = [x for x in possible_indices if not x in indices]
            indices_conf_scorer = random.sample(new_possible_indices, min(1000, len(new_possible_indices)))
            if len(new_possible_indices) < 1000:
                indices_conf_scorer = None
                logging.warning('We cannot train the scorer in this dataset.')
            else:
                X = []
                y = []
                for idx in tqdm(indices_conf_scorer):
                    example = dataset[idx]
                    X.append(example['acc_LM-nli_pred_ALL'] + [example['ragqa_ppl_utility']])
                    y.append(example['acc_LM'])
                X = np.array(X)
                y = np.array(y)
                X, y = shuffle(X, y, random_state=123)  
                # train model
                print('Training confidence model...')
                # more features, more estimators
                _n_estimators = 20
                param_grid = {'max_depth': [3, 4, 5], 'min_child_weight': [
                    1, 3, 5], 'gamma': [0], 'n_estimators': [_n_estimators], 'learning_rate': [0.1], 'subsample': [0.8], 'colsample_bytree': [0.8], 'reg_alpha': [1e-1, 1e-2], 'reg_lambda': [1e-1, 1e-2]}
                param_fit = {'eval_metric': 'auc', 'verbose': False,
                            'early_stopping_rounds': 1}
                model_builder = xgb.XGBRegressor(
                    random_state=123, booster='gbtree', objective='reg:logistic')

                conf_model = GridSearchCV(model_builder, param_grid,
                                        # fit_params=param_fit,
                                        cv=8, verbose=False, n_jobs=multiprocessing.cpu_count() - 1) #, scoring=make_scorer(spearmanr_scorer, greater_is_better=True))
                conf_model.fit(X, y)
                print('\nCONFIDENCE MODEL FITTED\n')

        p_true_avg_len = []
        it = 0
        for index in tqdm(indices):
            #if (it + 1 % 10) == 0: 
            gc.collect()
            torch.cuda.empty_cache()
            it += 1

            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            if args.format_ques:
               question = example['question']
               question = question[0].upper() + question[1:] + '?'          
            
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']

            full_responses = []

            # We sample one low temperature answer on which we will compute the
            # accuracy and args.num_generation high temperature answers which will
            # be used to estimate the entropy variants.

            if dataset_split == 'train' and args.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = args.num_generations + 1

            # we'll normally go through the else: just leave this original pice of code if want to use the HF model to predict here
            if not args.precomputed_gen:
                print('Generations are given, we no longer compute them here.')
                exit(0)
            else:
                most_likely_answer_dict = {
                            'response': example['predicted_answer'],
                            'token_log_likelihoods': example['token_log_likelihoods'],
                            'embedding': None, # for the moment!
                            'accuracy': example['acc_LM'] if args.acc_LM and 'acc_LM' in example.keys() else example['acc']}
                generations[example['id']].update({
                    'most_likely_answer': most_likely_answer_dict,
                    'reference': utils.get_reference(example)})
                # Append all predictions for this example to `generations`.
                full_responses = []
                for g, tl, in zip(example['full_responses_text'], example['full_responses_tlogp']):
                    full_responses.append((g, tl))
                generations[example['id']]['responses'] = full_responses
                accuracies.append(example['acc'])
                if args.compute_accuracy_at_all_temps:
                    for pred in full_responses:
                        accuracies.append(match(pred[0], correct_answer))


            if args.compute_p_true and dataset_split == 'validation':
                # Already compute p_true here. Avoid cost of generations in compute_uncertainty script.
                
                # get the example specific prompt (add knowledge and as many examples as possible)
                knowledge_len = len(model.tokenizer.encode(example['context']))
                few_shot_prompt = []
                for p, prompt_candidate in enumerate(p_true_few_shot_prompt):
                    prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
                    # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
                    # 200 buffer for question and 'Possible Answer'.
                    max_input_len = prompt_len + knowledge_len + num_generations * model.max_new_tokens + 200
                    
                    if max_input_len < model.token_limit:
                        few_shot_prompt.extend(prompt_candidate)
                    else:
                        print('Cutting of p_true prompt at length %d.', p)
                        break
                p_true_avg_len.append(p)

                few_shot_prompt.append(f"\n\nKnowledge:\n{example['context']}\n")
                few_shot_prompt = ''.join(few_shot_prompt)
                p_true = p_true_utils.calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in full_responses], few_shot_prompt,
                    hint=args.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)

            if args.compute_utility  and dataset_split == 'validation':
                if 'nli_utility_predicted' in example.keys():
                    utilitites_pred['nli_utility_predicted'].append(example['nli_utility_predicted'])      
                if 'acc-nli_utility_predicted' in example.keys():
                    utilitites_pred['acc-nli_utility_predicted'].append(example['acc-nli_utility_predicted'])             
                if 'acc_LM-nli_utility_predicted' in example.keys():
                    utilitites_pred['acc_LM-nli_utility_predicted'].append(example['acc_LM-nli_utility_predicted'])    
                if 'acc_LM_utility_predicted' in example.keys():
                    utilitites_pred['acc_LM_utility_predicted'].append(example['acc_LM_utility_predicted'])    
                if 'acc_utility_predicted' in example.keys():
                    utilitites_pred['acc_utility_predicted'].append(example['acc_utility_predicted'])                                                                                                      
                if 'mean_utility_predicted' in example.keys():
                    utilitites_pred['mean_utility_predicted'].append(example['mean_utility_predicted'])    
                if 'score_utility_predicted' in example.keys():
                    utilitites_pred['score_utility_predicted'].append(example['score_utility_predicted'])       
                if 'ragqa_ppl_utility' in example.keys():
                    utilitites_pred['ragqa_ppl_utility'].append(example['ragqa_ppl_utility'])    
                if 'ragqa_MSP_utility' in example.keys():
                    utilitites_pred['ragqa_MSP_utility'].append(example['ragqa_MSP_utility'])
                if 'ragqa_PMI_utility' in example.keys():
                    utilitites_pred['ragqa_PMI_utility'].append(example['ragqa_PMI_utility'])
                if 'ragqa_RenyiNeg_utility' in example.keys():
                    utilitites_pred['ragqa_RenyiNeg_utility'].append(example['ragqa_RenyiNeg_utility'])
                if 'ragqa_FisherRao_utility' in example.keys():
                    utilitites_pred['ragqa_FisherRao_utility'].append(example['ragqa_FisherRao_utility'])           

                if args.train_confidence and conf_model:
                    X_test = np.array(example['acc_LM-nli_pred_ALL'] + [example['ragqa_ppl_utility']])
                    y_pred = conf_model.predict(X_test)
                    utilitites_pred['utility_confidence_predicted'].append(y_pred[0])    
                                                 

        wandb.config.update(
            {'p_true_num_fewshot': np.mean(np.array(p_true_avg_len))}, allow_val_change=True)
        wandb.log(dict(len_p_true=np.mean(np.array(p_true_avg_len))))
        experiment_details['p_true_len'] = np.mean(np.array(p_true_avg_len))

        # Save generations for that split.
        utils.save(generations, f'{dataset_split}_generations.pkl')

        # Log overall accuracy.
        accuracy = np.mean(accuracies)
        print(f"Overall {dataset_split} split accuracy: {accuracy}")
        wandb.log({f"{dataset_split}_accuracy": accuracy})

        if dataset_split == 'validation':
            if args.compute_p_true:
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues],
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues],
                }
            if args.compute_utility:
                for key_ut in utilitites_pred.keys():
                    if utilitites_pred[key_ut]:
                        results_dict['uncertainty_measures'][key_ut] = utilitites_pred[key_ut]          


            print('Saving uncertainty measures...')
            utils.save(results_dict, 'uncertainty_measures.pkl')
            print('done')

    utils.save(experiment_details, 'experiment_details.pkl')
    logging.info('Run complete.')
    del model


if __name__ == '__main__':

    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()
    logging.info('Starting new run with args: %s', args)

    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    if args.compute_uncertainties:
        args.assign_new_wandb_id = False

    # First sample generations from LLM.
    logging.info('STARTING `generate_answers`!')
    main(args)
    logging.info('FINISHED `generate_answers`!')

    if args.compute_uncertainties:
        # Follow with uncertainty calculation script by default.
        args.assign_new_wandb_id = False
        gc.collect()
        torch.cuda.empty_cache()
        logging.info(50 * '#X')
        logging.info('STARTING `compute_uncertainty_measures`!')
        main_compute(args)
        logging.info('FINISHED `compute_uncertainty_measures`!')
