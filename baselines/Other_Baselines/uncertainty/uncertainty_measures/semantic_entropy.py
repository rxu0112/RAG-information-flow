"""Implement semantic entropy."""
import os
import pickle
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import wandb
import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Other_Baselines.uncertainty.models.huggingface_models import HuggingfaceModel
from Other_Baselines.uncertainty.utils import openai as oai
from Other_Baselines.uncertainty.utils import utils


DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("beberta_model")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "beberta_model").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction


class EntailmentLLM(BaseEntailment):

    entailment_file = 'entailment_cache.pkl'

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only

    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info('Restoring prediction cache from %s', entailment_cache_id)

        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir)

        with open(f'{wandb.run.dir}/{self.entailment_file}', "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        # Write the dictionary to a pickle file.
        utils.save(self.prediction_cache, self.entailment_file)

    def check_implication(self, text1, text2, example=None):
        if example is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, example['question'])

        logging.info('%s input: %s', self.name, prompt)

        hashed = oai.md5hash(prompt)
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            response = self.prediction_cache[hashed]
        else:
            if self.entailment_cache_only:
                raise ValueError
            response = self.predict(prompt, temperature=0.02)
            self.prediction_cache[hashed] = response

        logging.info('%s prediction: %s', self.name, response)

        binary_response = response.lower()[:30]
        if 'entailment' in binary_response:
            return 2
        elif 'neutral' in binary_response:
            return 1
        elif 'contradiction' in binary_response:
            return 0
        else:
            logging.warning('MANUAL NEUTRAL!')
            return 1

class EntailmentLlama:

    def __init__(self, model_path, device="cuda:0", tokenizer=None, model=None, batch_size=None):
        # self.name = name
        # self.model = HuggingfaceModel(
        #     name, stop_sequences='default', max_new_tokens=30)
        self.device = device
        self.prediction_cache = {}
        self.batch_size = batch_size or int(os.environ.get("SEMANTIC_ENTAILMENT_BATCH_SIZE", "16"))
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if model is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype
            ).to(device)
        else:
            self.model = model
        self.model.eval()

    def check_implication(self, text1, text2, example=None):
        if example is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, example['prompt'])
        if prompt in self.prediction_cache:
            response = self.prediction_cache[prompt]
        else:
            response = self.predict(prompt, temperature=0.02)
            self.prediction_cache[prompt] = response
        binary_response = response.lower()[:30]
        if 'entailment' in binary_response:
            return 2
        elif 'neutral' in binary_response:
            return 1
        elif 'contradiction' in binary_response:
            return 0
        else:
            logging.warning('MANUAL NEUTRAL!')
            return 1
    def equivalence_prompt(self, text1, text2, question):

        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"""
        prompt += "Response:"""

        return prompt

    def predict(self, prompt, temperature, max_new_tokens=10):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        predicted_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return predicted_answer

    def predict_batch(self, prompts, temperature=0.02, max_new_tokens=10):
        """Generate entailment labels for a batch of prompts."""
        if not prompts:
            return []

        responses = []
        for start in range(0, len(prompts), self.batch_size):
            prompt_batch = prompts[start:start + self.batch_size]
            inputs = self.tokenizer(
                prompt_batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
            prompt_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[:, prompt_len:]
            responses.extend(
                self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
        return responses

    def check_implication_batch(self, text_pairs, example=None):
        """Batch implication checks with prompt-level caching."""
        if example is None:
            raise ValueError
        prompts = [self.equivalence_prompt(text1, text2, example['prompt'])
                   for text1, text2 in text_pairs]

        predictions = [None] * len(prompts)
        uncached_prompts = []
        uncached_indices = []

        for idx, prompt in enumerate(prompts):
            if prompt in self.prediction_cache:
                predictions[idx] = self.prediction_cache[prompt]
            else:
                uncached_prompts.append(prompt)
                uncached_indices.append(idx)

        if uncached_prompts:
            uncached_predictions = self.predict_batch(uncached_prompts, temperature=0.02)
            for idx, response in zip(uncached_indices, uncached_predictions):
                self.prediction_cache[prompts[idx]] = response
                predictions[idx] = response

        result = []
        for response in predictions:
            binary_response = response.lower()[:30]
            if 'entailment' in binary_response:
                result.append(2)
            elif 'neutral' in binary_response:
                result.append(1)
            elif 'contradiction' in binary_response:
                result.append(0)
            else:
                logging.warning('MANUAL NEUTRAL!')
                result.append(1)
        return result


def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """strings_list: List of answers to cluster into semantic ids.
    example is the full prompt"""

    def are_equivalent_from_predictions(implication_1, implication_2):
        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
        return semantically_equivalent

    # Exact de-duplication preserves outputs while avoiding repeated entailment checks.
    unique_strings = []
    unique_index = {}
    original_to_unique = []
    for text in strings_list:
        key = text.strip()
        if key not in unique_index:
            unique_index[key] = len(unique_strings)
            unique_strings.append(text)
        original_to_unique.append(unique_index[key])

    # Initialise unique ids with -1.
    unique_semantic_ids = [-1] * len(unique_strings)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(unique_strings):
        # Check if string1 already has an id assigned.
        if unique_semantic_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            unique_semantic_ids[i] = next_id
            candidate_indices = list(range(i + 1, len(unique_strings)))
            if candidate_indices:
                if hasattr(model, 'check_implication_batch'):
                    implication_1 = model.check_implication_batch(
                        [(string1, unique_strings[j]) for j in candidate_indices], example=example)
                    implication_2 = model.check_implication_batch(
                        [(unique_strings[j], string1) for j in candidate_indices], example=example)
                else:
                    implication_1 = [
                        model.check_implication(string1, unique_strings[j], example=example)
                        for j in candidate_indices]
                    implication_2 = [
                        model.check_implication(unique_strings[j], string1, example=example)
                        for j in candidate_indices]

                for j, pred_1, pred_2 in zip(candidate_indices, implication_1, implication_2):
                    assert (pred_1 in [0, 1, 2]) and (pred_2 in [0, 1, 2])
                    if are_equivalent_from_predictions(pred_1, pred_2):
                        unique_semantic_ids[j] = next_id
            next_id += 1

    assert -1 not in unique_semantic_ids

    semantic_set_ids = [unique_semantic_ids[idx] for idx in original_to_unique]

    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy


def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy
