"""Compute uncertainty measures after generating answers — Flexible Version."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from Other_Baselines.uncertainty.uncertainty_measures.semantic_entropy import (
    get_semantic_ids, predictive_entropy, predictive_entropy_rao,
    cluster_assignment_entropy, logsumexp_by_id, EntailmentLlama,
)
from Other_Baselines.uncertainty.utils import eval


class UncertaintyMetrics:
    """Class to compute various uncertainty metrics."""

    def __init__(self, model_path, device="cuda:0"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.entailment_model = None

    def load_models(self, load_causal_lm=True, load_entailment=True):
        """Load required models based on selected metrics."""
        if load_causal_lm:
            print(f"Loading tokenizer and model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, torch_dtype=torch.bfloat16
            ).to(self.device)
            self.model.eval()
            print("✓ Model loaded")

        if load_entailment:
            print("Loading entailment model...")
            kwargs = {}
            if self.model is not None and self.tokenizer is not None:
                kwargs = {"tokenizer": self.tokenizer, "model": self.model}
            self.entailment_model = EntailmentLlama(self.model_path, self.device, **kwargs)
            print("✓ Entailment model loaded")

    def compute_p_true(self, question, most_probable_answer, brainstormed_answers,
                       few_shot_prompt=None, hint=False):
        """Calculate p_true uncertainty metric."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded.")

        prompt = (few_shot_prompt + '\n') if few_shot_prompt else ''
        q_start = question.find("Question:")
        prompt += question[q_start:] if q_start >= 0 else question
        prompt += 'Possible answer: ' + most_probable_answer + '\n'
        if not hint:
            prompt += 'Is the possible answer:\nA) True\nB) False\nThe possible answer is:'
        else:
            prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'

        return self._get_p_true_score(prompt)

    def _get_p_true_score(self, input_data):
        """Get the probability of the model answering A (True)."""
        device = next(self.model.parameters()).device
        input_data += ' A'
        tokenized = self.tokenizer(input_data, return_tensors='pt').to(device)['input_ids']
        target_ids = tokenized.clone()
        target_ids[0, :-1] = -100

        with torch.no_grad():
            loss = self.model(tokenized, labels=target_ids).loss

        if torch.isnan(loss):
            print("WARNING: NAN loss detected!")
            return float('nan')
        return -loss.item()

    def compute_perplexity(self, toklogprobs):
        """Compute perplexity from token log probabilities."""
        return eval.calculate_perplexity(toklogprobs)

    def compute_semantic_entropy(self, responses, log_liks, strict_entailment=False, example=None):
        """Compute semantic entropy and related metrics."""
        if self.entailment_model is None:
            raise ValueError("Entailment model not loaded.")

        semantic_ids = get_semantic_ids(
            responses, model=self.entailment_model,
            strict_entailment=strict_entailment, example=example)

        log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]
        log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')

        return {
            'cluster_assignment_entropy': cluster_assignment_entropy(semantic_ids),
            'regular_entropy': predictive_entropy(log_liks_agg),
            'semantic_entropy': predictive_entropy_rao(log_likelihood_per_semantic_id),
        }


def extract_log_liks(data):
    """Extract log likelihoods and answers from generation data."""
    toklogprobs_list, answer_list = [], []
    for g in data['generations']:
        answer_list.append(g['generation'])
        toklogprobs_list.append(g['toklogprobs'])
    return toklogprobs_list, answer_list


def auroc(y_true, y_score):
    """Calculate AUROC score, dropping non-finite scores."""
    pairs = [(yt, ys) for yt, ys in zip(y_true, y_score) if np.isfinite(ys)]
    if len(pairs) < len(y_score):
        print(f"WARNING: Dropped {len(y_score) - len(pairs)} non-finite scores.")
    if not pairs:
        raise ValueError("No finite scores available for AUROC computation.")

    yt = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    fpr, tpr, _ = metrics.roc_curve(yt, ys)
    return metrics.auc(fpr, tpr)


def main(args):
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        device = "cuda:0"
        print(f"Using GPUs: {args.gpus}")
    else:
        device = args.device

    print("=" * 60)
    print("Uncertainty Metrics Computation")
    print(f"P_true: {args.compute_p_true}  PPL: {args.compute_ppl}  SE: {args.compute_entropy}")
    print("=" * 60)

    # Load data
    data_all = torch.load(args.all_data, weights_only=False)
    hem = data_all['hem'].tolist()
    y_true = [0 if x < 0.5 else 1 for x in hem]
    pred = data_all['pred']

    data = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Total samples: {len(data)}")

    metrics_computer = UncertaintyMetrics(args.model_path, device=device)
    metrics_computer.load_models(
        load_causal_lm=args.compute_p_true or args.compute_ppl,
        load_entailment=args.compute_entropy,
    )

    results = {k: [] for k in ['P_true', 'PPL', 'regular_entropy', 'semantic_entropy']}

    print("\nProcessing examples...")
    for idx, sample in enumerate(tqdm(data, desc="Computing metrics")):
        first_answer = pred[idx]['first_sentence']
        question = sample["prompt"].replace("Answer:", "")

        log_liks, responses = extract_log_liks(sample)

        if args.compute_p_true:
            results['P_true'].append(
                metrics_computer.compute_p_true(question, first_answer, responses))

        if args.compute_ppl:
            results['PPL'].append(
                metrics_computer.compute_perplexity(sample['generations'][0]['toklogprobs']))

        if args.compute_entropy:
            ent = metrics_computer.compute_semantic_entropy(
                responses, log_liks, strict_entailment=args.strict_entailment, example=sample)
            results['regular_entropy'].append(ent['regular_entropy'])
            results['semantic_entropy'].append(ent['semantic_entropy'])

    # Compute AUROC
    print("\n" + "=" * 60)
    print("AUROC scores:")
    if args.compute_p_true:
        print(f"  P(true):           {auroc(y_true, results['P_true']):.4f}")
    if args.compute_ppl:
        print(f"  PPL:               {auroc(y_true, results['PPL']):.4f}")
    if args.compute_entropy:
        print(f"  Regular Entropy:   {auroc(y_true, [-x for x in results['regular_entropy']]):.4f}")
        print(f"  Semantic Entropy:  {auroc(y_true, [-x for x in results['semantic_entropy']]):.4f}")

    # Save
    save_dict = {"hem_scores": hem, "y_true": y_true}
    if args.compute_p_true:
        save_dict["p_true_scores"] = results['P_true']
    if args.compute_ppl:
        save_dict["ppl_scores"] = results['PPL']
    if args.compute_entropy:
        save_dict["regular_entropy_scores"] = [-x for x in results['regular_entropy']]
        save_dict["semantic_entropy_scores"] = [-x for x in results['semantic_entropy']]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute uncertainty metrics with flexible options")
    parser.add_argument("--dataset", default="...")
    parser.add_argument("--all_data", default="...")
    parser.add_argument("--output", default="...")
    parser.add_argument("--model_path", default="MODEL_PATH_PLACEHOLDER")
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--strict_entailment", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--compute_p_true", action="store_true")
    parser.add_argument("--compute_ppl", action="store_true")
    parser.add_argument("--compute_entropy", action="store_true")
    parser.add_argument("--compute_all", action="store_true")

    args = parser.parse_args()
    if args.compute_all:
        args.compute_p_true = True
        args.compute_ppl = True
        args.compute_entropy = True
    if not (args.compute_p_true or args.compute_ppl or args.compute_entropy):
        print("ERROR: No metrics selected! Use --compute_p_true, --compute_ppl, --compute_entropy, or --compute_all")
        exit(1)
    main(args)
