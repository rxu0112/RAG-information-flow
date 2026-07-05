
def read_NQ(item):
    
    if len(item["short_answers"])>1:
        print("More answers", len(item["short_answers"]), item["short_answers"])

    # add output for retrieval
    return {"question": item["question"], "answers": item["short_answers"], "q_id": item["example_id"], }

def read_TQA(item):
    
    # add output for retrieval
    return {"question": item["question"], "answers": item["answers"], "q_id": item["id"], }

def read_WebQ(item, idx):
    
    # add output for retrieval
    return {"question": item["question"], "answers": item["answers"], "q_id": idx, }


def read_SQuAD(item, idx):
    if len(item["answers"])>0:
        answers = item["answers"][0]["text"]
    else:
        answers = []

    # add output for retrieval
    return {"question": item["question"], "answers": answers, "context": item['context'],"q_id": idx, }

def read_RefuNQ(item, idx):
    
    # add output for retrieval
    return {"question": item["prompt"], 
            "answers": item["label"] if type(item["label"]) is list else [item["label"]], 
            "q_id": idx, }

def read_PopQA(item):
    
    # add output for retrieval
    return {"question": item["question"], "answers": item["answers"], "q_id": item["q_id"], } 




def get_entry_from_dataset(dataset_name, item, idx):
    if dataset_name=='NQ':
        return read_NQ(item)
    if dataset_name=='TQA':
        return read_TQA(item)
    if dataset_name=='WebQ':
        return read_WebQ(item, idx)
    if dataset_name=='SQuAD':
        return read_SQuAD(item, idx)
    if dataset_name=='RefuNQ':
        return read_RefuNQ(item, idx)   
    if dataset_name=='PopQA':
        return read_PopQA(item)       

