"""
Conditional probability of p(x2|x1). For each x1, there may be multiple x2 to be measured. Avoid double computing of x1.

About tokenization:
    - The first sequence should have the bos token (if any) but no eos token. Many models, e.g., LLaMA and GPT-2, do not automatically add eos tokens.
    - The second sequence should not have any special tokens.
    - Some tokenizers, e.g., GPT-2, treat words with or without space ahead differently. So the second sequence should begain with a space if there is a space.
"""
from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizer
def causal_lm_log_probs(
    model,
    tokenizer:PreTrainedTokenizer,
    seq1_text: str,
    seq2_text_list: List[str],
    verbose = False
):
    """
    calculate the logprob of each input-output pair.
    """
    if seq1_text[-1] == ' ':
        raise ValueError((
            "If the input sentence ends with a space, ",
            "the tokenization of seq1 may produce an additional token. "
            "Then seq1_length is increased by 1, "
            "which is to spilit the whole sentence into seq1 and seq2.")
        )
    seq1_len = len(tokenizer(seq1_text).input_ids)
    device = model.device

    # Join seq1 and seq2 and split them after tokenization
    join_text = [seq1_text + ' ' + t for t in seq2_text_list]
    join_ids_list = [
        tokenizer(t, return_tensors = 'pt').input_ids.to(device) 
                     for t in join_text]
    seq1_ids = join_ids_list[0][..., :seq1_len]
    seq2_ids_list = [k[..., seq1_len:] for k in join_ids_list]

    with torch.no_grad():
        seq1_outputs = model(input_ids = seq1_ids, return_dict = True)
    seq1_logits = seq1_outputs.logits
    seq1_key_values = seq1_outputs.past_key_values

    logprobs_list = []
    for seq2_ids in seq2_ids_list:
        with torch.no_grad():
            seq2_logits = model(input_ids = seq2_ids,
                                past_key_values = seq1_key_values,return_dict = True).logits

        extended_logits = torch.concat(
            [seq1_logits[...,-1:,:], seq2_logits[...,:-1,:]], dim = -2
        ).view(-1, seq1_logits.shape[-1])
        seq2_logprobs = - F.cross_entropy(extended_logits, 
                                        seq2_ids.view(-1), 
                                        reduction = 'none')
        logprobs_list.append(seq2_logprobs)
    
    if verbose:
        # print the first one
        for seq2_ids, logps in zip(seq2_ids_list, logprobs_list):
            prev_ids = seq1_ids[..., -1:].view(-1).tolist() + \
                        seq2_ids[...,:-1].view(-1).tolist()
            next_ids = seq2_ids.view(-1).tolist()
            prev_tokens = tokenizer.convert_ids_to_tokens(prev_ids)
            next_tokens = tokenizer.convert_ids_to_tokens(next_ids)
            print('Log probabilities: ')
            for ptoken, ntoken, logp in zip(prev_tokens, next_tokens, logps.tolist()):
                print(f'{ptoken} -> {ntoken}: {logp}')
    
    return logprobs_list
        
def causal_lm_log_probs_join(
    model,
    tokenizer:PreTrainedTokenizer,
    seq1_text: str,
    seq2_text: str,
    verbose = False
):
    """
    Concatenate seq1 and seq2 and output logits jointly
    """
    device = model.device
    input_ids = tokenizer(seq1_text + ' ' + seq2_text, return_tensors='pt').input_ids.to(device)
    # [1, seq_len]
    seq1_len = len(tokenizer(seq1_text).input_ids)

    logits = model(input_ids).logits
    seq2_logits = logits[...,seq1_len-1:-1,:].view(-1, logits.shape[-1])
    seq2_ids = input_ids[..., seq1_len:].view(-1)
    logprobs = - F.cross_entropy(seq2_logits, seq2_ids, reduction = 'none')

    if verbose:
        tot_len = input_ids.shape[-1]
        seq2_len = tot_len - seq1_len
        print(f'Total: {tot_len}, Seq1: {seq1_len}, Seq2: {seq2_len}')
        prev_tokens = tokenizer.convert_ids_to_tokens(input_ids[..., seq1_len-1:-1].view(-1))
        next_tokens = tokenizer.convert_ids_to_tokens(seq2_ids)
        for ptoken, ntoken, logp in zip(prev_tokens, next_tokens, logprobs.tolist()):
            print(f'{ptoken} -> {ntoken}: {logp}')

    return logprobs

def effect_of_insert_one_space(model_path):
    """
    To decode "China" and " China" using LLaMA
    """
    # model_path = '/data1/llm/llama2_hf/llama-2-7b'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map = {"": 0},
        torch_dtype = 'auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = model.device
    seq1_ids = tokenizer('The giant panda is the national animal of',
                         return_tensors = 'pt').input_ids.to(device)
    next_token_ids = tokenizer(' China', 
                               add_special_tokens=False, 
                               return_tensors = 'pt').input_ids.to(device)

    def show_next_token(input_ids, past_key_values):
        with torch.no_grad():
            outs = model(input_ids, past_key_values = past_key_values)
        logits, past_kv = outs.logits, outs.past_key_values
        probs = F.softmax(logits[0, -1, :], dim = -1)
        rank = torch.argsort(probs, descending=True)
        rank_p = probs[rank]
        for i in range(20):
            token= tokenizer.convert_ids_to_tokens([rank[i]])[0]
            print(f'\t{token}: {rank_p[i].item()}')
    
        return logits, past_kv
    # Q -> "China"
    print(' '.join(tokenizer.convert_ids_to_tokens(seq1_ids[0])))
    logits, past_kv = show_next_token(seq1_ids, None)

    # " " -> "China"
    seq1_sp_ids = torch.concat([seq1_ids, next_token_ids[...,:1]], dim = -1)
    print(' '.join(tokenizer.convert_ids_to_tokens(seq1_sp_ids[0])))

    _ = show_next_token(next_token_ids[:,:1], past_kv)
    print('--------Whole seq')
    _ = show_next_token(seq1_sp_ids, None)


def speed_test(model, tokenizer):
    seq1_text = 'The giant panda is the national animal'*10
    seq2_text = ['of 2 Chinese provinces. They live in forests.'] * 20
    import time
    _ = causal_lm_log_probs(model, tokenizer, seq1_text, seq2_text)
    print('Start key_value calculate')
    st = time.time()
    for i in range(10):
        with torch.no_grad():
            _ = causal_lm_log_probs(model, tokenizer, seq1_text, seq2_text)
    dur = time.time() - st
    print(dur)

    print('Start whole calculate')
    st = time.time()
    for i in range(10):
        with torch.no_grad():
            for t in seq2_text:
                _ = causal_lm_log_probs_join(model, tokenizer, seq1_text, t)
    dur = time.time() - st
    print(dur)


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path') # gpt2, ...
    parser.add_argument('--n_gpu', type = int, default = 1)
    args = parser.parse_args()

    # effect_of_insert_one_space(args.model_path)
    # exit()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map = 'auto',
        torch_dtype = 'auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    speed_test(model, tokenizer)
    exit()

    seq1_text = 'The giant panda is the national animal of'
    seq2_text_list = ['China', 'the United States']
    print('Calculate log_probs based on key_values of input')
    logprobs = causal_lm_log_probs(model, tokenizer, seq1_text, seq2_text_list, True)
    print('Calculate log_probs of the whole sequence')
    logprobs_join = causal_lm_log_probs_join(model, tokenizer, seq1_text, seq2_text_list[0], True)

    p1 = torch.exp(logprobs[0].sum())
    p2 = torch.exp(logprobs_join.sum())
    print((f'key_value result: {p1.item()}\n'
           f'join result: {p2.item()}'))
    