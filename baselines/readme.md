**Baseline Experiments**

(1). `baselines/baseline_data_prepare.py`

Prepares and processes the data for baseline methods.

(2). Baseline methods

Main script for running the baseline methods. The generated outputs are stored in results/model_dataset/baselines.

`Attention Score`. An attention-based uncertainty estimator using LM-Polygraph's `AttentionScore` to quantify model confidence from attention weight distributions.

`Focus`. A logit-attribution-based uncertainty estimator from LM-Polygraph that identifies important tokens via IDF-weighted focus scores.

`Embedding Model`. A training-based calibrator that extracts LLM hidden states and trains an XGBoost classifier to predict answer correctness. This method is referred to as "KnowingMore" in the paper.

`Utility Ranker`. A training-based passage utility estimator using Contriever retrieval and a BERT ranker to assess contextual relevance.

`Other Baselines`. A multi-round generation pipeline that computes black-box uncertainty metrics (semantic entropy and predictive entropy) and a verbalized uncertainty quantification method P(True).

(3) collect_baselines.py 

A script for aggregating uncertainty scores from all baseline methods into a unified evaluation format. Please refer to `readme_collect.md` for details.

