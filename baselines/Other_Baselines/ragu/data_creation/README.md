
## Datasets and Format

We download NQ, TriviaQA, SQuAD, WebQuestions from the DPR repository [here](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py). Note that Contriever github uses a slightly different (answers tokenisation) version of NQ and TriviaQA that can be download [here (FiD github)](https://github.com/facebookresearch/FiD/blob/main/get-data.sh), but we do not use it, see if needed.

We standarise each dataset to have the same .jsonl format throughout our code. So first you should run [create_retrieval_data.py](./create_retrieval_data.py) or any script you like to generate an input file from your chosen dataset that has the following required fields (note that your file could also have other fields, they will not be used):

```
{'question': QUESTION, "answers": ANSWERS, "q_id": QID}
```
```ÀNSWERS``` is a list of strings, the other two a string.


## Passage Retrieval

We follow retrieval steps and scripts from [Self-RAG](https://github.com/AkariAsai/self-rag/tree/main?tab=readme-ov-file#retriever-setup).
Follow instructions from Contriever [here](https://github.com/facebookresearch/contriever/tree/main). 

```
git clone https://github.com/facebookresearch/contriever.git
```

By default, we use [Contriever](https://github.com/facebookresearch/contriever) as our retrieval component.

### Download data

Download preprocessed passage data used in DPR.

```
cd retrieval_lm
wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
```

Then, download the generated passages. We use [Contriever-MSMARCO](https://huggingface.co/facebook/contriever-msmarco) version.

```
wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
```

### Run retriever

You can run passage retrieval by running the command below.

```
python passage_retrieval.py
    --model_name_or_path facebook/contriever-msmarco 
    --passages psgs_w100.tsv
    --passages_embeddings "wikipedia_embeddings/*"
    --data YOUR_INPUT_FILE
    --output_dir YOUR_OUTPUT_FILE
    --n_docs 20
```

Your input file should be the .jsonl file standarised with the fields as described above. The output file will be the input file augmented with retrieved passages. That is, for each question a field will be added named ```"ctxs"``` that will contain a list of passages, each passage element will be a dictionary with: ```{"id": ID, "title": TITLE, "text":TEXT, "score": SCORE, "hasanswer": HASANSWER}```.

Note: this is a requirement to run contriever code: ```pip install faiss-cpu --no-cache```.

Note: when running the contriever-msmarco, it gives a warning message about 'pooler' weights/bias not loaded. We found from the retriever code and the description of the architecture in [Contriever's paper](https://openreview.net/forum?id=jKN1pXi7b0) that these are in BERTModel but not used by Contriever (its pooling is based on averaging the last hidden states). In [this post](https://github.com/huggingface/transformers/issues/14017#issue-1027173952) it seems that could be a bug/always given warning if the flag ```add_pooling_layer=False``` is not passed properly.
