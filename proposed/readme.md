We first generate predictions from LLM models and evaluate the correctness of the predictions.

1. `proposed/inference.py` generates predictions on datasets based on the selected inference model.

2. `proposed/eval_hem_answerable.py` evaluates if the predicted answers are correct.

Shapley value computation to determine which context tokens are useful for answering questions.

1. `proposed/gemma_tokenization_diff.py` obtains samples that are tokenized differently between Llama and Gemma, because Shapely values are computed based on the basic token unit. This strategy is time-saving compared to recompute Shapley values for all Gemma tokenized samples.

2. `proposed/SHAP_Qwen3_8B.py`, `proposed/SHAP_MiniLM_L12.py`, and `proposed/SHAP_BGE-v2-m3.py` estimate the ground-truth relevance of each context token

3. `proposed/gemma_shap_replacement.py` uses the tokenization differences to merge the Shapley result of Llama and the "replacement" result of Gemma
