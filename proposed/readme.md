**We first generate predictions from LLM models and evaluate the correctness of the predictions.**

(1). `proposed/inference.py`

Generates predictions on datasets based on the selected inference model.

(2). `proposed/eval_hem_answerable.py` 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Evaluates if the predicted answers are correct.

**Then we compute Shapley values to identify which context tokens are most relevant for answering a question.**

(1). `proposed/gemma_tokenization_diff.py`

Identifies samples that are tokenized differently between LLaMA-3.2-3B-Instruct and Gemma-3-4B-it. Since Shapley values are computed at the token level, this script avoids recomputing Shapley values for all Gemma-tokenized samples, significantly reducing computation time.

(2). `proposed/SHAP_Qwen3_8B.py`, `proposed/SHAP_MiniLM_L12.py`, and `proposed/SHAP_BGE-v2-m3.py`

Estimate the ground-truth relevance score of each context token. Because LLaMA-3.2-3B-Instruct and LLaMA-3-8B-Instruct share the same tokenizer, the relevance estimation computed for one model can be directly reused for the other.

(3). `proposed/gemma_shap_replacement.py`

Uses the identified tokenization differences to combine the Shapley results from LLaMA with the corresponding replacement results for Gemma.
