"""Compute p_true uncertainty metric."""
import logging

import string
import re

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def match(prediction, ground_truth):
    # do not know why the base code did not normalise the answer in this metric, while it does for exact match
    prediction = normalize_answer(prediction)
    for gt in ground_truth:
        if normalize_answer(gt) in prediction:
            return 1
    return 0

def construct_few_shot_prompt(
        *, model, dataset, indices, prompt, brief, brief_always, make_prompt,
        num_generations, metric):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info('P_TRUE >> Current Question: '.ljust(25) + current_question)

        responses = []
        for j in range(num_generations + 1):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            response, _, _ = model.predict(local_prompt, temperature)
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)

            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example['answers']['text']]
                logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']

        prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        # 200 buffer for question and 'Possible Answer'.
        max_input_len = prompt_len + num_generations * model.max_new_tokens + 200

        if max_input_len < model.token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning('Cutting of p_true prompt at length %d.', it)
            break

    return ''.join(few_shot_prompt), all_responses, it


def construct_few_shot_prompt_RAG_exec(
        *, model, dataset, indices, prompt, num_generations, metric, fewshot_prompt_rag):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        #current_question = make_prompt(context, question, None, brief, brief_always)
        #local_prompt = prompt + current_question
        local_prompt = prompt.format_map({"instruction":example["question"], "paragraph": example["context"], "fewshots": fewshot_prompt_rag}) 
        logging.info('P_TRUE >> Current Question: '.ljust(25) + local_prompt)

        responses = []
        for j in range(num_generations + 1):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            response, _, _ = model.predict(local_prompt, temperature)
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)

            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                #is_correct = metric(response, example, model)
                is_correct = match(response, example['answers']['text'])
                answers = [answer for answer in example['answers']['text']]
                logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']

        prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        # 200 buffer for question and 'Possible Answer'.
        max_input_len = prompt_len + num_generations * model.max_new_tokens + 200

        if max_input_len < model.token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning('Cutting of p_true prompt at length %d.', it)
            break

    return ''.join(few_shot_prompt), all_responses, it

def construct_few_shot_prompt_RAG_given(model, in_context_exs, num_generations, acc_LM):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    for it, example in enumerate(in_context_exs):
        prompt_candidate = []
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        #local_prompt = prompt.format_map({"instruction":example["question"], "paragraph": example["context"], "fewshots": fewshot_prompt_rag}) 
        logging.info('P_TRUE >> Current Question: '.ljust(25) + question)
        response = example['predicted_answer']
        logging.info('P_TRUE >> Current Response: '.ljust(25) + response)
        prompt_candidate += [f'{response.strip()} \n']
        responses = [response] + example['full_responses_text']   
        for s_response in  example['full_responses_text']:
            prompt_candidate += [f'{s_response.strip()} \n']

        # Save most likely response and compute correctness metric for it.
        most_likely_response = response
        is_correct = example['acc_LM'] if acc_LM and 'acc_LM' in example.keys() else example['acc']
        answers = [answer for answer in example['answers']['text']]
        logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
        logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        all_responses[it] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']

        prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        # 200 buffer for question and 'Possible Answer'.
        max_input_len = prompt_len + num_generations * model.max_new_tokens + 200

        if max_input_len < model.token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning('Cutting of p_true prompt at length %d.', it)
            break
    #print('\n***************\n')
    #print(f'token_limit: {model.token_limit} / shots: {it}')
    #print(''.join(few_shot_prompt))
    #exit()

    return ''.join(few_shot_prompt), all_responses, it

def construct_few_shot_prompt_RAG_given_list(model, in_context_exs, num_generations, acc_LM):
    """Construct few shot prompt for p_true uncertainty metric.
    Samples for the prompt are pre-computed (given). It returns a list of all the examples instead of concat and chunking."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    for it, example in enumerate(in_context_exs):
        prompt_candidate = []
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        #local_prompt = prompt.format_map({"instruction":example["question"], "paragraph": example["context"], "fewshots": fewshot_prompt_rag}) 
        logging.info('P_TRUE >> Current Question: '.ljust(25) + question)
        response = example['predicted_answer']
        logging.info('P_TRUE >> Current Response: '.ljust(25) + response)
        prompt_candidate += [f'{response.strip()} \n']
        responses = [response] + example['full_responses_text']   
        for s_response in  example['full_responses_text']:
            prompt_candidate += [f'{s_response.strip()} \n']

        # Save most likely response and compute correctness metric for it.
        most_likely_response = response
        is_correct = example['acc_LM'] if acc_LM and 'acc_LM' in example.keys() else example['acc']
        answers = [answer for answer in example['answers']['text']]
        logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
        logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        all_responses[it] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']

        #prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        ## At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        ## 200 buffer for question and 'Possible Answer'.
        #max_input_len = prompt_len + num_generations * model.max_new_tokens + 200
        #
        #if max_input_len < model.token_limit:
        #    few_shot_prompt.extend(prompt_candidate)
        #else:
        #    logging.warning('Cutting of p_true prompt at length %d.', it)
        #    break

        few_shot_prompt.append(prompt_candidate)


    #return ''.join(few_shot_prompt), all_responses, it    
    return few_shot_prompt, all_responses, it    

def calculate_p_true(
        model, question, most_probable_answer, brainstormed_answers,
        few_shot_prompt, hint=False):
    """Calculate p_true uncertainty metric."""

    if few_shot_prompt:
        prompt = few_shot_prompt + '\n'
    else:
        prompt = ''

    prompt += 'Question: ' + question
    prompt += '\nBrainstormed Answers: '
    for answer in brainstormed_answers + [most_probable_answer]:
        prompt += answer.strip() + '\n'
    prompt += 'Possible answer: ' + most_probable_answer + '\n'
    if not hint:
        prompt += 'Is the possible answer:\n'
        prompt += 'A) True\n'
        prompt += 'B) False\n'
        prompt += 'The possible answer is:'
    else:
        prompt += 'Do the brainstormed answers match the possible answer? Respond with A if they do, if they do not respond with B. Answer:'

    log_prob = model.get_p_true(prompt)

    return log_prob
