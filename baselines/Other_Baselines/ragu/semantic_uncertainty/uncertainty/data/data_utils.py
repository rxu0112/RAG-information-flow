"""Data Loading Utilities."""
import os
import json
import hashlib
import datasets
import sys
sys.path.append('../utils')
from utils import load_jsonlines, load_file
from datasets import DatasetDict
import os.path
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_dataset_dict(dataset, proportion, most_likely_file_name_template, 
                    generations_file_name_template, 
                    original_file_name_template,
                    top_n, split,  
                    utilities_file_name_template=None, ood_train_dataset=None):
    """ Note when split is 'train' it contains only the subset used for few-shots on p(true). Not the full train set. """
    dataset_dict = {
        "question": [],
        "answers": [],
        "context": [],
        "predicted_answer": [],
        "token_log_likelihoods": [],
        "acc": [],
        "acc_LM": [],      
        "full_responses_text": [],
        "full_responses_tlogp": [],
        "id": []
    }

    if utilities_file_name_template:
        dataset_dict.update({"ragqa_ppl_utility": [], "ragqa_nll_utility": [],
            "ragqa_MSP_utility": [], "ragqa_PMI_utility": [], "ragqa_RenyiNeg_utility": [], 
            "ragqa_FisherRao_utility": []}) 

    available_train_data = True
    mlfn = most_likely_file_name_template.replace('SPLIT', split)
    mlfn = mlfn.replace('DATA', ood_train_dataset) if (ood_train_dataset and split == 'train') \
                        else mlfn.replace('DATA', dataset)
    print('Uploading Most Likely Answers file ...  ', mlfn)
    if split == 'train' and not os.path.isfile(mlfn): 
        print('Train Most Likely Answer file not found. Generate training data for p(true) first.', mlfn)
        exit(0)
    
    input_data = load_file(mlfn)
    input_data = input_data[:int(len(input_data)*proportion)]
    print(f'Taken {proportion}% of the data: {len(input_data)}')

    gsfn = generations_file_name_template.replace('SPLIT', split)
    gsfn = gsfn.replace('DATA', ood_train_dataset) if (ood_train_dataset and split == 'train') \
                        else gsfn.replace('DATA', dataset)
    if split == 'train' and not os.path.isfile(gsfn): 
        print('No samples from training data found. Generate training samples for p(true) first.')
        exit(0)
    else:
        print('Uploading samples file ...  ', gsfn)
        input_data_samples = load_file(gsfn)
        print(f'Sample data: {len(input_data_samples)}')
        assert len(input_data) == len(input_data_samples)        

    if utilities_file_name_template:
        ufn = utilities_file_name_template.replace('SPLIT', split)
        ufn = ufn.replace('DATA', ood_train_dataset) if (ood_train_dataset and split == 'train') \
                        else ufn.replace('DATA', dataset)
        print('Uploading utilities file ...  ', ufn)
        input_context_utilities = load_file(ufn)
        input_context_utilities = input_context_utilities[:int(len(input_context_utilities)*proportion)]
        print(f'Taken {proportion}% of the sample data: {len(input_context_utilities)}')        
    else:
        print('We do not need utilities.')
        input_context_utilities = [{'q_id':x['q_id']} for x in input_data] ## dummy this so the loop below stays the same
    
    ofn = original_file_name_template.replace('SPLIT', split)
    ofn = ofn.replace('DATA', ood_train_dataset) if (ood_train_dataset and split == 'train') \
                        else ofn.replace('DATA', dataset)
    print('Uploading file for contexts ...  ', ofn)
    ori_contexts = load_file(ofn)
    if split == 'train':
        train_indices = [y["q_id"] for y in input_data]
        ori_contexts = [x for x in ori_contexts if x["q_id"] in train_indices]

    for i, x in enumerate(ori_contexts):
        retrieval_result = x["ctxs"][:top_n]
        evidences = ["[{}] ".format(
            i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
        input_data[i]["paragraph"] = "\n".join(evidences)

    for item, item_samples, item_utilities in zip(input_data, input_data_samples, input_context_utilities):
        assert item["q_id"] == item_samples["q_id"]
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]
        
        dataset_dict["question"].append(item["instruction"])
        dataset_dict["context"].append(item["paragraph"])
        if not type(item["golds"]) is list:
            dataset_dict["answers"].append({'text': [item["golds"]]})
        else:    
            dataset_dict["answers"].append({'text': item["golds"]})

        dataset_dict["predicted_answer"].append(item["output"])
        dataset_dict["token_log_likelihoods"].append(item["toklogprob"])
        dataset_dict["acc"].append(item["acc"])
        dataset_dict["acc_LM"].append(item["acc_LM"])
        dataset_dict["full_responses_text"].append(
                        [gen["generation"] for gen in item_samples["generations"]])
        dataset_dict["full_responses_tlogp"].append(
                        [gen["toklogprobs"] for gen in item_samples["generations"]])

        if utilities_file_name_template:
            assert len(item_utilities["ctxs"]) > 0
            item_contexts = item_utilities["ctxs"][:top_n]
            item_contexts_keys = item_contexts[0].keys()

            # baseline utilities
            dataset_dict["ragqa_nll_utility"].append(float(item["nll"]))
            dataset_dict["ragqa_ppl_utility"].append(float(item["ppl"]))
            dataset_dict["ragqa_MSP_utility"].append(float(item["MSP"]))
            dataset_dict["ragqa_PMI_utility"].append(float(item["PMI"]))
            dataset_dict["ragqa_RenyiNeg_utility"].append(float(item["RenyiNeg"]))
            dataset_dict["ragqa_FisherRao_utility"].append(float(item["FisherRao"]))

            # just add those utilities that are present
            preference_rank = ['rl-nli_pred', 'rl_pred', 'nli_pred', 'acc-nli_pred', 'acc_LM-nli_pred', 
                            'acc_pred', 'acc_LM_pred', 'acc-ties_pred','mean', 'score']
            for pr in preference_rank:
                if pr in item_contexts_keys:  
                    prk = pr.split('_pred')[0]
                    if not prk+'_utility_predicted' in dataset_dict.keys():
                        dataset_dict[prk+'_utility_predicted'] = []
                    dataset_dict[prk+'_utility_predicted'].append(max([float(ctx[pr]) for ctx in item_contexts]))

            # add all predicted utilitites (i.e. for each input passage)
            if 'acc_LM-nli_pred' in item_contexts_keys:
                if not 'acc_LM-nli_pred_ALL' in dataset_dict.keys():
                    dataset_dict['acc_LM-nli_pred_ALL'] = []
                dataset_dict['acc_LM-nli_pred_ALL'].append([float(ctx[pr]) for ctx in item_contexts])


        dataset_dict["id"].append(item["q_id"])

    return dataset_dict

def load_ds(dataset_name, seed, add_options=None):
    """Load dataset."""
    user = os.environ['USER']

    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')
        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}}

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        scratch_dir = os.getenv('SCRATCH_DIR', '.')
        path = f"{scratch_dir}/{user}/semantic_uncertainty/data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question['exact_answer'], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question['exact_answer']
                    ]
                else:
                    exact_answers = [question['exact_answer']]

                dataset_dict["answers"].append({
                    "text": exact_answers,
                    "answer_start": [0] * len(question["exact_answer"])
                })
            else:
                dataset_dict["answers"].append({
                    "text": question["ideal_answer"],
                    "answer_start": [0]
                })
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)

        # Split into training and validation set.
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    else:
        raise ValueError

    return train_dataset, validation_dataset

def load_ds_precomputed(dataset_name, 
                        eval_mode,
                        proportion,
                        most_like_file_name_template, 
                        samples_file_name_template, 
                        original_file_name_template,
                        top_n, 
                        utilities_file_name_template=None, ood_train_dataset=None): #get_training_set_generations_most_likely_only,
    """Load dataset with pre-computed most likely generation and samples."""

    train_dataset, validation_dataset = None, None
    if dataset_name in ['tqa', 'squad', 'webq', 'nq', 'popqa', 'popqa3k', 'refunq', 'ambigqa']:

        ds_train = datasets.Dataset.from_dict(
            get_dataset_dict(dataset_name, proportion, most_like_file_name_template, 
            samples_file_name_template, 
            original_file_name_template,
            top_n, 
            'train', ood_train_dataset=ood_train_dataset))

        ds_dev = datasets.Dataset.from_dict(
            get_dataset_dict(dataset_name, proportion, most_like_file_name_template, 
            samples_file_name_template, 
            original_file_name_template,
            top_n, 
            eval_mode,
            utilities_file_name_template=utilities_file_name_template))

        # Split into training and validation set.
        dataset = DatasetDict({
        'train': ds_train,
        'valid': ds_dev
            }) 
        train_dataset = dataset['train']
        validation_dataset = dataset['valid']
    else:
        raise ValueError

    return train_dataset, validation_dataset