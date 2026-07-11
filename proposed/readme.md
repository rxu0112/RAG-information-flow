**We first generate predictions from LLM models and evaluate the correctness of the predictions.**

(1). `proposed/inference.py`

Generates predictions on datasets based on the selected inference model.

(2). `proposed/eval_hem_answerable.py` 

Evaluates if the predicted answers are correct.

---
**Then we compute Shapley values to identify which context tokens are most relevant for answering a question.**

(1). `proposed/gemma_tokenization_diff.py`

Since Shapley values are computed at the token level, this script avoids recomputing Shapley values for all samples that are tokenized identically between LLaMA and Gemma, significantly reducing computation time.

(2). `proposed/SHAP_Qwen3_8B.py`, `proposed/SHAP_MiniLM_L12.py`, and `proposed/SHAP_BGE-v2-m3.py`

Estimate the ground-truth relevance score of each context token. Because LLaMA-3.2-3B-Instruct and LLaMA-3-8B-Instruct share the same tokenizer, the relevance estimation computed for one model can be directly reused for the other.

(3). `proposed/gemma_shap_replacement.py`

Uses the identified tokenization differences to combine the Shapley results from LLaMA with the corresponding replacement results for Gemma.

---
**Uncertainty Quantification via the Proposed Method**

(1). `proposed/llama_3B_wash_samples.py`, `proposed/llama_8B_wash_samples.py`, and `proposed/gemma_4B_wash_samples.py`

Filter out low-quality samples, such as instances where marker words cannot be identified or the generated predictions are incomplete.

(2) `Ours` folder

Main script for the proposed uncertainty quantification pipeline. The generated outputs are stored in results/model_dataset/loop_bf16.

(3) `proposed/result_preprocess.py`

Preprocesses the generated results by removing unqualified samples, computing Rank-Biased Overlap (RBO) and KL divergence with respect to the uniform distribution, and splitting the data into training, validation, and test sets.

(4)`proposed/calibrator.py`

Uses XGBoost to train a calibrator that maps the multi-dimensional uncertainty representation to a scalar uncertainty score.
