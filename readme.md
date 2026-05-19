# Information Flow Reveals When to Trust Language Models
This is the source code for the paper [Information Flow Reveals When to Trust Language Models](icml.cc/virtual/2026/poster/60884).
![screenshot](Method.png)
**(a)** An example of a short-form, information-seeking QA. **(b)** Principal information flow is extracted in reverse from the model's complete information flow. The resulting Emergence Order records the sequence of input tokens added to this principal flow, with earlier tokens indicating greater importance for the final generation. For clarity, we neglect MLP operations as they operate independently on each token. **(c)** The contribution of each input token to the next generated token is defined as the sum of all valid paths from itself to the last input token embedding in the final layer.  Contribution Layout represents the contributions of all input tokens.

We first generate predictions from LLM models and evaluate the correctness of the predictions.

1. proposed/inference.py generate predictions on datasets based on selected inference model.

2. proposed/eval_hem_answerable.py evaluate the if the predicted answer is correct.

Shaply value computation to determine which context tokens are useful for answering questions.

1. proposed/gemma_tokenization_diff.py obtainS samples which are tokenized differently between llama and gemma, because shaply values are computated based on the basic token unit. This stratygy is time saving than recompute shaply values for all gemma tokenized samples.

2. proposed/SHAP_Qwen3_8B.py, proposed/SHAP_MiniLM_L12.py, and proposed/SHAP_BGE-v2-m3.py estimate the ground-truth relevnce of each context tokens

3. proposed/gemma_shap_replacement.py uses the tokenization differences to merge the shapley result of llama and the "replacement" result of gemma

Uncertainty Quantification via the Proposed Method

1. proposed/llama_3B_wash_samples.py, proposed/llama_8B_wash_samples.py, and proposed/gemma_4B_wash_samples.py filter out low-quality samples, such as samples who can not find mark words or have incomplete predictions.

2. here is our main script (ChenYi), output to results/model_dataset/loop_bf16

3. proposed/result_preprocess.py preprocesses data by removing unqualified samples, computing RBO and KL divergence with uniform, and split data into train, validation, and test files. 

4. proposed/calibrator.py uses XGBoost to train a calibrator to map the multi-dimensional UQ representation to scalar.

Conduct Baselines

1. baselines/baseline_data_prepare.py processes data for baselines.

2. here is our baseline script (ChenYi), output to results/model_dataset/baselines, (change rebuild name)

Main Experiment Result

1. proposed/plot.py evaluates the performance of baselines and the proposed method

OOD Ablation

1. ablation/ood_test.py evaluate the generalization ability of a trained calibrator on different datasets.

Calibrated Baselines

1. ablation/baselines_calibrator.py evaluates the performance of calibrated 1d baselines

Bias from Rankers

1. ablation/humans_vs_reranker.py compares the simulatbility from original and human verified samples on Llama-3.2-3B-Instruct.

Faithfulness Ablation

1. ablation/inference_faithfulness.py ablates top-ranked and low-ranked tokens to do inference.

2. ablation/eval_faithfulness.py evaluates if the predicted answers are correct after ablating top-ranked and low-ranked tokens.

3. ablation/faithfulness_comparison.py compares the results between top- and low-rank token ablation

Concentration from Ranker

1. ablation/result_preprocess_ranker_concentration.py quantifies the concentration of relevance scoresf from the ranker.

2. ablation/calibrator_ranker_concentration.py trains calibrators based on relevance-based concentration.








