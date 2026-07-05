# Uncertainty Quantification in Retrieval Augmented Question Answering

This repository contains data and code for our paper [Uncertainty Quantification in Retrieval Augmented Question Answering](). Please contact me at lperez@ed.ac.uk for any question.


- [data creation](data_creation/) this folder contains scripts to pre-process different QA datasets and standarised them to the format that our code requires.
- [retrieval augmented question answering](retrieval_qa/) scripts to run retrieval augmented QA with different backbone LLMs to generate answers, collect information necessary for different uncertainty estimation methods, and compute RAG accuracy. It also contains scripts to collect target QA model judgements of passage utility to train the Passage Utility predictor. Pieces of code adapted from baseilnes in [(Asai et al, 2024)](https://github.com/AkariAsai/self-rag).
- [Passage Utility](passage_utility/) contains the code for the Passage Utility predictor, code based on [Fang et al., 2024](https://github.com/edwinrobots/BayesianOpt_uncertaiNLP2024).
- [semantic uncertainty quantification](semantic_uncertainty/) code to run comparison uncertainty estimation methods and evaluation.


Our experiments are run with this docker image:
```
docker pull lauhaide/algomo:v1
```
