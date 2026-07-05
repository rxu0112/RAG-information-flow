import jsonlines
import json
from vllm import SamplingParams

# look at chat-template with RAG from here: https://cookbook.openai.com/examples/using_logprobs
# Answer the question in no more than five words. Context: {context} Question: {question} Answer:
PROMPT_DICT = {
        "chat_directRagQA_REAR2Llama": { "system": ("Answer the question in no more than five words."
            ),
        "user": (
        "Context:\n"
        "{paragraph}\n\n"
        "Question: {instruction} Answer:"
        )},
        "chat_directRagQA_REAR2gemma": { "system": ("Answer the question in no more than five words."
            ),
        "user": (
        "Context:\n"
        "{paragraph}\n\n"
        "Question: {instruction} Answer:"
        )},
    "prompt_no_input_retrieval_SELFRAG": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),   
    "chat_no_input_retrieval_SELFRAG": { "system": ("You are a helpful, respectful and honest assistant. "
            ),
        "user": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}"
        )
    },      
    "chat_no_input_retrieval_SELFRAG2": { "system": ("You are a helpful, respectful and honest assistant. "
            ),
        "user": (
        "Write a short phrase that appropriately completes the request.\n\n"
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}"
        )
    },          
    "prompt_directRagQA_REAR": (
        "Knowledge:\n"
        "{paragraph}\n\n"
        "Answer the following question with a very short phrase, such as \"1998\", \"May 16th, 1931\", or \"James Bond\", "
        "to meet the criteria of exact match datasets. \n\n"
        "Question: {instruction}\n\nAnswer: "
    ),
    "chat_directRagQA_REAR": { "system": ("You are a helpful, respectful and honest assistant. "
            ),
        "user": (
        "Knowledge:\n"
        "{paragraph}\n\n"
        "Answer the following question with a very short phrase, such as \"1998\", \"May 16th, 1931\", or \"James Bond\", "
        "to meet the criteria of exact match datasets. \n\n"
        "Question: {instruction}"
        )
    },
    "chat_directRagQA_REAR2": { "system": ("You are a helpful, respectful and honest assistant. "
            ),
        "user": (
        "Knowledge:\n"
        "{paragraph}\n\n"
        "Answer the following question with a very short phrase, such as \"1998\", \"May 16th, 1931\", \"James Bond\", "
        "or \"Barack Obama and Joe Biden\", to meet the criteria of exact match datasets. \n\n"
        "Question: {instruction}"
        )
    },       
    "chat_directRagQA_REAR3": { "system": ("You are a helpful assistant. "
            ),
        "user": (
        "Knowledge:\n"
        "{paragraph}\n\n"
        "Answer the following question with a very short phrase.\n\n"
        "Question: {instruction}"
        )
    },      
    "chat_directRagQA_REAR3Llama": { "system": ("You are a helpful assistant. "
        "Answer the user question with a very short phrase. "
            ),
        "user": (
        "Knowledge:\n"
        "{paragraph}\n\n"
        "Question: {instruction}"
        )
    },        
    "chat_directRagQA_REAR4": { "system": ("You are a helpful, respectful and honest assistant. "
            ),
        "user": (
        "Knowledge:\n"
        "{paragraph}\n\n"
        "Given these passages, answer the following question with a very short phrase."
        "Before even answering the question, consider whether you have sufficient information in the passages to answer the question fully.\n\n"
        "Question: {instruction}"
        )
    },       
    "prompt_noRAG_REAR": (
        "Answer the following question with a very short phrase, such as \"1998\", \"May 16th, 1931\", or \"James Bond\", "
        "to meet the criteria of exact match datasets. \n\n"
        "Question: {instruction}\n\nAnswer: "
    ),    
    "chat_noRAG_REAR3":  { "system": ("You are a helpful assistant. "
            ),
        "user": (
        "Answer the following question with a very short phrase.\n\n"
        "Question: {instruction}"
        )
    },      
    "chat_noRAG_REAR3Llama":  { "system": ("You are a helpful assistant. "
        "Answer the following question with a very short phrase. "
            ),
        "user": (
        "Question: {instruction}"
        )
    },     
    "prompt_directRagQA_RECOMP": (
        "{fewshots}"
        "{paragraph}\n\n"
        "{instruction}\n"
        "Answer:"
    ),
    "chat_directRagQA_RECOMP": { "system": ("You are a helpful, respectful and honest assistant. "
            ),
        "user": ("You should JUST provide a concise answer as in the following examples.\n\n"
        "who won a million on deal or no deal\n"
        "assistant: Tomorrow Rodriguez\n\n"
        "who is the woman washing the car in cool hand luke\n"
        "assistant: Joy Harmon\n\n"
        "who is the actor that plays ragnar on vikings\n"
        "assistant: Travis Fimmel\n\n"
        "who said it's better to have loved and lost\n"
        "assistant: Alfred , Lord Tennyson\n\n"
        "name the first indian woman to be crowned as miss world\n"
        "assistant: Reita Faria\n\n"
        "{paragraph}\n{instruction}")
    },
    "prompt_noRAG_RECOMP": {'user':
        # "{fewshots}"
        "{instruction}\n",
        # "Answer:"
    },
    "prompt_utility_judgement_retrieval": (
        "Below is an instruction that describes a task. "
        "Read the paragraph and instruction and then judge whether the paragraph is relevant to generate the response. "
        "Your judgement should be yes, no or partially.\n\n "
        "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:\n{answers}\n\n### Your Judgement:"
    ),         
    "prompt_context_relevance_ares": (
        "Given the following question and document, you must analyze the provided document "
        "and determine whether it is sufficient for answering the question. In your evaluation, "
        "you should consider the content of the document and how it relates to the provided question. "
        "Output your final verdict by strictly following this format: \"[[Yes]]\" if the document is "
        "sufficient and \"[[No]]\" if the document provided is not sufficient. Do not provide any "
        "additional explanation for your decision. \n\nQuestion: {instruction} "
        "\n\nDocument: {paragraph} "
    ),         
    "prompt_answer_faithfulness_ares": (
        "Given the following question, document, and answer, you must analyze the provided answer "
        "and determine whether it is faithful to the contents of the document. The answer must not "
        "offer new information beyond the context provided in the document. The answer also must "
        "not contradict information provided in the document. Output your final verdict by strictly "
        "following this format: \"[[Yes]]\" if the answer is faithful to the document and \"[[No]]\" if "
        "the answer is not faithful to the document. Do not provide any additional explanation for your "
        "decision. \n\nQuestion: {instruction} \n\nDocument: {paragraph} \n\nAnswer: {answers} "
    ),
    "prompt_accuracy_eval": (
"You need to check whether the prediction of a question-answering system to a question is correct. You should make the judgment based on a list of "
"ground truth answers provided to you. Your response should be \"correct\" if the prediction is correct or \"incorrect\" if the prediction is wrong.\n"
"\nQuestion: Who authored The Taming of the Shrew (published in 2002)?"
"\nGround truth: [\"William Shakespeare\", \"Roma Gill\"]"
"\nPrediction: W Shakespeare"
"\nCorrectness: correct\n"
"\nQuestion: Who authored The Taming of the Shrew (published in 2002)?"
"\nGround truth: [\"William Shakespeare\", \"Roma Gill\"]"
"\nPrediction: Roma Gill and W Shakespeare"
"\nCorrectness: correct\n"
"\nQuestion: Who authored The Taming of the Shrew (published in 2002)?"
"\nGround truth: [\"William Shakespeare\", \"Roma Gill\"]"
"\nPrediction: Roma Shakespeare"
"\nCorrectness: incorrect\n"
"\nQuestion: What country is Maharashtra Metro Rail Corporation Limited located in?"
"\nGround truth: [\"India\"]"
"\nPrediction: Maharashtra"
"\nCorrectness: incorrect\n"
"\nQuestion: What's the job of Song Kang-ho in Parasite (2019)?"
"\nGround truth: [\"actor\"]"
"\nPrediction: He plays the role of Kim Ki-taek, the patriarch of the Kim family."
"\nCorrectness: correct\n"
"\nQuestion: Which era did Michael Oakeshott belong to?"
"\nGround truth: [\"20th-century philosophy\"]"
"\nPrediction: 20th century."
"\nCorrectness: correct\n"
"\nQuestion: Edward Tise (known for Full Metal Jacket (1987)) is in what department?"
"\nGround truth: [\"sound department\"]"
"\nPrediction: 2nd Infantry Division, United States Army"
"\nCorrectness: incorrect\n"
"\nQuestion: What wine region is Finger Lakes AVA a part of?"
"\nGround truth: [\"New York wine\"]"
"\nPrediction: Finger Lakes AVA"
"\nCorrectness: incorrect\n"
"\nQuestion: {instruction}"
"\nGround truth: {answers}"
"\nPrediction: {output}"
"\nCorrectness:"
    ),
"chat_accuracy_eval-rlhf-calib": {
    "system": ("You are a helpful, respectful and honest assistant. "),
    "user": (
        "Are the following two answers to my question Q semantically equivalent?"
        "\n\nQ: {instruction}"
        "\nA1: {answers}"
        "\nA2: {output}"
        "\n\nPlease answer with a single word, either \"Yes.\" or \"No.\""
        )
    },  
"chat_accuracy_eval-mine": {
    "system": ("You are a helpful, respectful and honest assistant. "),
    "user": (
        "You need to check whether the prediction of a question-answering system to my question Q is correct. You should make the judgment based on a list of "
        "ground truth answers provided to you. Your response should be \"correct\" if the prediction is correct or \"incorrect\" if the prediction is wrong.\n"
        "\n\nQ: {instruction}"
        "\nGround truth: {answers}"
        "\nPrediction: {output}"
        )
    },  
}
#, and explain your reasoning.
#"\n\nPlease answer with a single word, either \"correct\" or \"incorrect\"."


def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst


def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data


def save_file_jsonl(data, fp):
    with jsonlines.open(fp, mode='w') as writer:
        writer.write_all(data)


def postprocess_answers_closed(output, task, choices=None):
    final_output = None
    if choices is not None:
        for c in choices.split(" "):
            if c in output:
                final_output = c
    #if task == "fever" and output in ["REFUTES", "SUPPORTS"]:
    #    final_output = "true" if output == "SUPPORTS" else "REFUTES"
    #if task == "fever" and output.lower() in ["true", "false"]:
    #    final_output = output.lower()
    if final_output is None:
        return output
    else:
        return final_output        


def getChatMessages(model_name, prompt_name, item):
    if 'Meta-Llama-3.1' in model_name:
        return  [
                    { "role" : "system",
                      "content" : PROMPT_DICT[prompt_name]["system"]
                    },
                    { "role" : "user",
                      "content": PROMPT_DICT[prompt_name]["user"].format_map(item)
                    }
                ]
    elif 'gemma-2' in model_name or 'Mistral' in model_name:
        return  [
                    { "role" : "user",
                      "content": PROMPT_DICT[prompt_name]["user"].format_map(item)
                    }
                ] 

STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', '$\n\n', '#\n\n', '+\n\n', '*\n\n', '$\n', '#\n', '+\n', '*\n', '/\n', '/\n']

def call_model(prompts, model, args, tokenizer, seed=None):
    """ Note on sampling params vLLM vs HF transformers inference params: https://github.com/vllm-project/vllm/discussions/539"""
    # We set those we need, others are left default as in:
    # https://docs.vllm.ai/en/latest/dev/sampling_params.html
    model_bos_id = tokenizer.bos_token_id
    model_eos_id = tokenizer.eos_token_id
    stop = []
    if args.do_stop:
        stop = STOP_SEQUENCES
    stop += [tokenizer.eos_token]
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=args.max_new_tokens, logprobs=args.logprobs, seed=seed, stop=stop)
    preds = model.generate(prompts, sampling_params, use_tqdm=False)
    tmp = preds

    toklogdists = [] # time-step token distributions
    toklogprobs = [] # most likely sequence logprobs
    toklogprobs_lm = [] # logprobs for most likely sequence w/o context/prefix, i.e., just forward the completion and get logprobs
    tokids = [] # token ids
    toks = [] # decoded tokens
    # the above 5 lists should have the same number of elements (special care when post-processing completions), and decoded tokens should be same as .text output attribute
    sectoklogprob = [] # the second most likely token at first decoding position (Guillem)
    for pred in preds:
        # logprobs for tokens in generated output
        assert len(pred.outputs[0].logprobs) == len(pred.outputs[0].token_ids)
        pred_lp = [] 
        pred_dist = [] 
        pred_toks = []
        pred_tok_ids = []
        first_pos = True
        flase_start = False
        for i, lp in enumerate(pred.outputs[0].logprobs):
            lp_values = list(lp.values())
            if not args.chat_template or args.do_stop:
                if lp_values[0].decoded_token.strip() == '' and i == 0:
                    flase_start = True
                    continue ##just continue the fist decoded token is a white space
                stop_token = None
                for s in STOP_SEQUENCES:
                    if s in lp_values[0].decoded_token:
                        stop_token = s
                        break
                if stop_token:
                    isnn = lp_values[0].decoded_token.split(stop_token) # needed for pre-trained models' outputs (i.e., no chat-tunned)
                    if len(isnn) == 2: # and not isnn[0] and not isnn[1]: 
                        if i == 0 or flase_start: 
                            flase_start = False
                            continue  ##just continue the fist decoded token contains \n\n
                        else: 
                            if isnn[0]: pred_toks.append(isnn[0])
                            break                 
                else:
                    flase_start = False
            pred_lp.append(lp_values[0].logprob)
            # pred_dist中存储每个时间步的所有token的logprob，用于计算熵等不确定性指标
            pred_dist.append([v.logprob for v in lp_values])
            pred_toks.append(lp_values[0].decoded_token)
            pred_tok_ids.append(pred.outputs[0].token_ids[i])
            # Guillem's first token metric (instead of token dist just take the difference between top and second-top more likely tokens at first time step)
            if first_pos:
                if args.logprobs > 1:
                    sectoklogprob.append(lp_values[1].logprob)
                    first_pos = False
                else:
                    sectoklogprob.append(None) 

        #generated string by Llama3.1-8B-Instruct has this token at the end '' or linebreak, the .text attribute does not have this token correspondance. hope this one occurs only at the end!
        # other models also??
        if len(pred_tok_ids) > 0 and pred_tok_ids[-1] == model_eos_id:
            pred_lp = pred_lp[:-1]
            pred_dist = pred_dist[:-1]
            pred_toks = pred_toks[:-1]
            tokids.append(pred_tok_ids[:-1])
        else:
            tokids.append(pred_tok_ids)
        toklogprobs.append(pred_lp) 
        # token distributions of generated output
        toklogdists.append(pred_dist)
        toks.append(pred_toks)
    
    # final text post-processing
    if not args.chat_template or args.do_stop:
        strip_preds = []
        for i, pred in enumerate(preds):
            p = pred.outputs[0].text.strip()
            for s in STOP_SEQUENCES:
                if p.startswith(s):
                    p = p.split(s)[1]
                    break
            for s in STOP_SEQUENCES:
                if s in p:
                    p = p.split(s)[0]
            strip_preds.append(p.strip())
        preds = strip_preds
    else:
        preds = [pred.outputs[0].text for pred in preds]
    # double check we take the correct toks/probs, etc. corresponding to the final text
    # for i, p in enumerate(preds):
    #     assert p.strip() == ("".join(toks[i])).strip(), "we should take same text attribute and decoded tokens, p:{} -- toks:{} -- \n{}".format(p, toks[i], tmp[i])
    # double check consistency between model text and token decoding
    # for i, p in enumerate(preds):
    #     decoded = tokenizer.decode(tokids[i], skip_special_tokens=True)
        # if p.strip() != decoded.strip():
        #     # 不再抛错，打印警告方便调试
        #     print(f"⚠️ Warning: mismatch detected!\n"
        #         f"- preds text   : {p.strip()}\n"
        #         f"- decoded toks : {decoded.strip()}\n")

    postprocessed_preds = [postprocess_output(pred) for pred in preds]

    if args.compute_pmi:
        # token distributions of unconditioned generated output
        # do we need a specific chat-template for chat-tunned LLMs? which format?
        #lm_prompts = build_lm_prompts(postprocessed_preds)
        sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        max_tokens=1, prompt_logprobs=1, skip_special_tokens=False)
        # shall we put this in a template when using instruction-tuned models?
        # i.e., to be an assistant shaped prompt: "<|start_header_id|>assistant<|end_header_id|>\n\nLinda Davis.<|eot_id|>" ?
        #messages = []
        #for pp in postprocessed_preds:
        #    messages.append([
        #        { "role" : "assistant",
        #            "content" : pp
        #        }
        #    ])
        #tokenizer = model.get_tokenizer()
        #tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True    
        #lm_preds = model.generate(preds, sampling_params, use_tqdm=False)
        
        #for i, lm_pred in enumerate(lm_preds):
        for i, seq_tokids in enumerate(tokids):
            lm_preds = model.generate({'prompt_token_ids': [model_bos_id] + list(seq_tokids)}, sampling_params, use_tqdm=False)
            lm_pred = lm_preds[0]
            lm_pred_dist = []
            # skip first lp is None for input token 128000 which is '' ??!! added by the tokenizer.
            lm_pred_prompt_logprobs = lm_pred.prompt_logprobs[1:] if lm_pred.prompt_token_ids[0] == model_bos_id else lm_pred.prompt_logprobs
            lm_pred_prompt_token_ids = lm_pred.prompt_token_ids[1:] if lm_pred.prompt_token_ids[0] == model_bos_id else lm_pred.prompt_token_ids
            assert lm_pred_prompt_token_ids == list(seq_tokids), "lm_pred:{} pred:{} \n\n {}\n{} ".format(lm_pred_prompt_token_ids, list(seq_tokids), lm_preds, tmp)
            for lp, tid in zip(lm_pred_prompt_logprobs, lm_pred_prompt_token_ids):
                lm_pred_dist.append(lp[tid].logprob)
            toklogprobs_lm.append(lm_pred_dist)
    

    # toklogprobs could be obtained from toklogdists_pred, but depends on dec algo (as we use greedy should be strightforward)
    # but just better extract here...
    return postprocessed_preds, toklogprobs, sectoklogprob, toklogdists, toklogprobs_lm

    # how to get embeddings:
    #https://github.com/vllm-project/vllm/issues/1654 --- not ok code
    #https://github.com/vllm-project/vllm/discussions/310
    #https://github.com/vllm-project/vllm/issues/4435

def postprocess_output(pred):
    pred = pred.replace("</s>", "")

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred                