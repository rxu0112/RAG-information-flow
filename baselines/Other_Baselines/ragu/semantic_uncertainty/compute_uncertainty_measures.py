"""Compute uncertainty measures after generating answers."""
from collections import defaultdict
import logging
import os
import pickle
import numpy as np
import wandb

from Other_Baselines.ragu.semantic_uncertainty.analyze_results import analyze_run
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.data.data_utils import load_ds
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import get_semantic_ids
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import logsumexp_by_id
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import predictive_entropy_rao
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import cluster_assignment_entropy
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import context_entails_response
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT35
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import EntailmentGPT4Turbo
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures.semantic_entropy import EntailmentLlama
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.uncertainty_measures import p_true as p_true_utils
from Other_Baselines.ragu.semantic_uncertainty.uncertainty.utils import utils


utils.setup_logger()

EXP_DETAILS = 'experiment_details.pkl'

REFUSE_PHRASE = [
    'this information is not provided in',
    'this information is not available in',
    'this information is not in the provided text',
    'not specified in the text',
    'the passage does not state',
    'the passage does not say',
    'the answer is not provided in the text',
    'not provided in the text',
    'this document does not say',
    'this question cannot be answered from the provided text',
    'there is no information about',
    'the passage does not specify',
    'not specified in the given text',
    'there is no information provided about',
    'there is no specific',
    'not available', 
    'not provided',
    'cannot be answered',
    'does not contain the answer',
    "couldn't find any information"]


def main(args):

    if args.train_wandb_runid is None:
        args.train_wandb_runid = args.eval_wandb_runid

    #user = os.environ['USER']
    scratch_dir = '/mnt/data/obqa/outputs'
    wandb_dir = f'{scratch_dir}/uncertainty'
    pod_jobid = os.getenv('POD_NAME')
    project = "semantic_uncertainty" if not args.debug else "semantic_uncertainty_debug"
    if args.assign_new_wandb_id:
        logging.info('Assign new wandb_id.')
        api = wandb.Api()
        old_run = api.run(f'{args.restore_entity_eval}/{project}/{args.eval_wandb_runid}')
        wandb.init(
            entity=args.entity,
            project=project,
            dir=wandb_dir,
            notes=f'pod_id: {pod_jobid}, experiment_lot: {args.experiment_lot}',
            # For convenience, keep any 'generate_answers' configs from old run,
            # but overwrite the rest!
            # NOTE: This means any special configs affecting this script must be
            # called again when calling this script!
            config={**old_run.config, **args.__dict__},
        )

        def restore(filename):
            old_run.file(filename).download(
                replace=True, exist_ok=False, root=wandb.run.dir)

            class Restored:
                name = f'{wandb.run.dir}/{filename}'

            return Restored
    else:
        logging.info('Reuse active wandb id.')

        def restore(filename):
            class Restored:
                name = f'{wandb.run.dir}/{filename}'
            return Restored

    if args.train_wandb_runid != args.eval_wandb_runid:
        logging.info(
            "Distribution shift for p_ik. Training on embeddings from run %s but evaluating on run %s",
            args.train_wandb_runid, args.eval_wandb_runid)

        is_ood_eval = True  # pylint: disable=invalid-name
        api = wandb.Api()
        old_run_train = api.run(f'{args.restore_entity_train}/semantic_uncertainty/{args.train_wandb_runid}')
        filename = 'train_generations.pkl'
        old_run_train.file(filename).download(
            replace=True, exist_ok=False, root=wandb.run.dir)
        with open(f'{wandb.run.dir}/{filename}', "rb") as infile:
            train_generations = pickle.load(infile)
        wandb.config.update(
            {"ood_training_set": old_run_train.config['dataset']}, allow_val_change=True)
    else:
        is_ood_eval = False  # pylint: disable=invalid-name

    wandb.config.update({"is_ood_eval": is_ood_eval}, allow_val_change=True)

    # Load entailment model.
    if args.compute_predictive_entropy:
        logging.info('Beginning loading for entailment model.')
        if args.entailment_model == 'deberta':
            entailment_model = EntailmentDeberta()
        elif args.entailment_model == 'gpt-4':
            entailment_model = EntailmentGPT4(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-3.5':
            entailment_model = EntailmentGPT35(args.entailment_cache_id, args.entailment_cache_only)
        elif args.entailment_model == 'gpt-4-turbo':
            entailment_model = EntailmentGPT4Turbo(args.entailment_cache_id, args.entailment_cache_only)
        elif 'llama' in args.entailment_model.lower():
            entailment_model = EntailmentLlama(args.entailment_cache_id, args.entailment_cache_only, args.entailment_model)
        else:
            raise ValueError
        logging.info('Entailment model loading complete.')


    if args.recompute_accuracy:
        # This is usually not enabled.
        logging.warning('Recompute accuracy enabled. This does not apply to precomputed p_true!')
        metric = utils.get_metric(args.metric)

    # Restore outputs from `generate_answrs.py` run.
    result_dict_pickle = restore('uncertainty_measures.pkl')
    with open(result_dict_pickle.name, "rb") as infile:
        result_dict = pickle.load(infile)
    result_dict['semantic_ids'] = []

    validation_generations_pickle = restore('validation_generations.pkl')
    with open(validation_generations_pickle.name, 'rb') as infile:
        validation_generations = pickle.load(infile)

    entropies = defaultdict(list)
    validation_embeddings, validation_is_true, validation_answerable, validation_is_refuse = [], [], [], []
    p_trues = []
    msl = []
    avg_ans_len = []
    count = 0  # pylint: disable=invalid-name

    def is_answerable(generation):
        return len(generation['reference']['answers']['text'][0]) > 0 and \
                not (generation['reference']['answers']['text'][0].strip().lower() == "nec") # RefuNQ has this value for unanswarable questions.

    def is_refuse(generation):
        for s in REFUSE_PHRASE:
            if s in generation["most_likely_answer"]["response"].strip().lower() : 
                return True
        return False
 
    # Loop over datapoints and compute validation entropies.
    for idx, tid in enumerate(validation_generations):

        example = validation_generations[tid]
        if args.format_ques:
            question = example['question']
            question = question[0].upper() + question[1:] + '?'
        else:
           question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        msl.append(len(most_likely_answer['response'].strip().split()))
        avg_ans_len.append(sum([len(x[0].strip().split()) for x in full_responses])/len(full_responses))

        if not args.use_all_generations:
            if args.use_num_generations == -1:
                raise ValueError
            responses = [fr[0] for fr in full_responses[:args.use_num_generations]]
        else:
            responses = [fr[0] for fr in full_responses]

        if args.recompute_accuracy:
            logging.info('Recomputing accuracy!')
            if is_answerable(example):
                acc = metric(most_likely_answer['response'], example, None)
            else:
                acc = 0.0  # pylint: disable=invalid-name
            validation_is_true.append(acc)
            logging.info('Recomputed accuracy!')

        else:
            validation_is_true.append(most_likely_answer['accuracy'])

        validation_is_refuse.append(is_refuse(example))
        validation_answerable.append(is_answerable(example))
        validation_embeddings.append(most_likely_answer['embedding'])
        logging.info('validation_is_true: %f', validation_is_true[-1])

        if args.compute_predictive_entropy:
            # Token log likelihoods. Shape = (n_sample, n_tokens)
            if not args.use_all_generations:
                log_liks = [r[1] for r in full_responses[:args.use_num_generations]]
            else:
                log_liks = [r[1] for r in full_responses]

            for i in log_liks:
                assert i

            if args.compute_context_entails_response:
                # Compute context entails answer baseline.
                entropies['context_entails_response'].append(context_entails_response(
                    context, responses, entailment_model))

            if args.condition_on_question and args.entailment_model == 'deberta':
                responses = [f'{question} {r}' for r in responses]

            # Compute semantic ids.
            semantic_ids = get_semantic_ids(
                responses, model=entailment_model,
                strict_entailment=args.strict_entailment, example=example)

            result_dict['semantic_ids'].append(semantic_ids)

            # Compute entropy from frequencies of cluster assignments.
            entropies['cluster_assignment_entropy'].append(cluster_assignment_entropy(semantic_ids))

            # Length normalization of generation probabilities.
            log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

            # Compute naive entropy.
            entropies['regular_entropy'].append(predictive_entropy(log_liks_agg))

            # Compute semantic entropy.
            log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
            pe = predictive_entropy_rao(log_likelihood_per_semantic_id)
            entropies['semantic_entropy'].append(pe)

            # pylint: disable=invalid-name
            log_str = 'semantic_ids: %s, avg_token_log_likelihoods: %s, entropies: %s'
            entropies_fmt = ', '.join([f'{i}:{j[-1]:.2f}' for i, j in entropies.items()])
            # pylint: enable=invalid-name
            logging.info(80*'#')
            logging.info('NEW ITEM %d at id=`%s`.', idx, tid)
            logging.info('Context:')
            logging.info(example['context'])
            logging.info('Question:')
            logging.info(question)
            logging.info('True Answers:')
            logging.info(example['reference'])
            logging.info('Low Temperature Generation:')
            logging.info(most_likely_answer['response'])
            logging.info('Low Temperature Generation Accuracy:')
            logging.info(most_likely_answer['accuracy'])
            logging.info('High Temp Generation:')
            logging.info([r[0] for r in full_responses])
            logging.info('High Temp Generation:')
            logging.info(log_str, semantic_ids, log_liks_agg, entropies_fmt)

        count += 1
        if count >= args.num_eval_samples:
            logging.info('Breaking out of main loop.')
            break

    logging.info('Accuracy on original task: %f', np.mean(validation_is_true))
    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    result_dict['validation_is_false'] = validation_is_false
    logging.info('False prop on validation: %f', np.mean(validation_is_false))

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    result_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    result_dict['validation_is_refuse'] = validation_is_refuse
    logging.info('Refuse prop on validation: %f', np.mean(validation_is_refuse))

    if 'uncertainty_measures' not in result_dict:
        result_dict['uncertainty_measures'] = dict()

    if args.compute_predictive_entropy:
        result_dict['uncertainty_measures'].update(entropies)

    # naive baseline based on answer length, not normalised but delimited
    result_dict['uncertainty_measures']['MSL'] = msl
    result_dict['uncertainty_measures']['AVG_ANS_LEN'] = avg_ans_len

    utils.save(result_dict, 'uncertainty_measures.pkl')

    if args.compute_predictive_entropy:
        entailment_model.save_prediction_cache()

    if args.analyze_run:
        # Follow up with computation of aggregate performance metrics.
        logging.info(50 * '#X')
        logging.info('STARTING `analyze_run`!')
        analyze_run(wandb.run.id)
        logging.info(50 * '#X')
        logging.info('FINISHED `analyze_run`!')


if __name__ == '__main__':
    parser = utils.get_parser(stages=['compute'])
    args, unknown = parser.parse_known_args()  # pylint: disable=invalid-name
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    logging.info("Args: %s", args)

    main(args)
