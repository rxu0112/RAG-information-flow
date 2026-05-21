**Main Experimental Results**

We first evaluate the performance of the baseline methods and the proposed approach.

(1). `evaluation/plot.py`
Computes AUROC and AUPRC across different methods and plots the corresponding AUROC and AUPRC curves. It also reports the accuracy and Expected Calibration Error (ECE) of calibration-based methods. The results correspond to Table 1, Table 3, Table 7, Table 10, Figure 7, and the left and middle columns of Table 6.

---
**Ablation Studies**

We further conduct extensive ablation studies to analyze different components of the proposed framework.

(1). `evaluation/ood_test.py`

Evaluates the generalization ability of calibrators trained on one dataset when applied to entirely different datasets. The results are reported in Table 2.

(2). `evaluation/baselines_calibrator.py`

Evaluates calibrated baseline methods for fair comparison, demonstrating that the post-hoc calibrator itself is not the primary source of performance improvement. Detailed discussions and results are provided in Appendix K.

(3). `evaluation/humans_vs_reranker.py`

Validates the relevance layouts estimated by Qwen3-Reranker-8B and measures the associated bias through human annotation. The comparison is conducted under the simulatability metrics of LLaMA-3.2-3B-Instruct, with results shown in Table 8 and Table 9.

(4). `evaluation/inference_faithfulness.py`

Performs token ablation experiments by removing top-ranked and low-ranked tokens based on contribution layouts and emergence order, followed by inference on the modified samples.

(5). `evaluation/eval_faithfulness.py`

Evaluates whether the predicted answers remain correct after ablating top-ranked and low-ranked tokens.

(6). `evaluation/faithfulness_comparison.py`

Measures the drop in AUROC and AUPRC after token ablation, demonstrating the causal faithfulness of the proposed method. The results are presented in Figure 6.

(7). `evaluation/result_preprocess_ranker_concentration.py`

Quantifies the concentration of relevance scores for context tokens produced by ranking models.

(8) `evaluation/calibrator_ranker_concentration.py`

Trains calibrators using relevance-based concentration features, showing that relevance layouts from rankers alone cannot fully capture how language models process contextual information. The corresponding results are reported in the right column of Table 6.
