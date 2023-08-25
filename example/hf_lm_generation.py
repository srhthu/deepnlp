"""
Example to show how to use CausalLM to generate.
"""
# %%
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, GPT2Tokenizer, BloomTokenizerFast
# %%
def id2token(x):
    return tokenizer.convert_ids_to_tokens(x)
# %%
model_path = "/data1/llm/llama2_hf/llama-2-7b"
# model_path = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map = {"": 0}, 
    torch_dtype= 'auto'
)
# %%
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 0
# %%
prompt = "Panda's favorite food is" # length 12
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
def target_tokenize(text, add_eos = True):
    token_ids = tokenizer(text, add_special_tokens=False).input_ids 
    if add_eos:
        token_ids += [tokenizer.eos_token_id]
    return token_ids

def batch_target_tokenize(text, add_eos = True):
    batch_ids = [target_tokenize(k, add_eos = add_eos) for k in text]
    seq_lens = [len(k) for k in batch_ids]
    max_len = max(seq_lens)
    batch_ids = [k + [0] * (max_len - len(k)) for k in batch_ids]
    return torch.tensor(batch_ids), torch.tensor(seq_lens)

options = ["bamboo",
           "dragon fruit"]
#options_ids = tokenizer(options, truncation = True, max_length=512, 
#                        padding = 'longest',return_tensors="pt").input_ids.cuda()
options_ids = [torch.tensor(target_tokenize(k)).unsqueeze(0).cuda() for k in options]
options_ids_bs, options_lengths = batch_target_tokenize(options)
# %%
gen_outputs = model.generate(input_ids, max_new_tokens = 100, do_sample = True,
                             top_k = 50, top_p = 0.8, length_penalty = 10.0)
print(tokenizer.batch_decode(gen_outputs, skip_special_tokens=True))
print(id2token(gen_outputs[0][len(input_ids[0]):]))
# %%
# First get seq1 key_values, then get seq2 probs
seq1_outputs = model(input_ids = input_ids,
                return_dict = True,
                output_hidden_states = True)
# %%
print('logits: ', seq1_outputs.logits.shape) # [1, 12, 32000]
past_kv = seq1_outputs.past_key_values 
# (num_layer, 2), each k,v: [bs, n_head, seq_len, head_dim]
print('key: ', past_kv[0][0].shape)
print('last hidden_states', seq1_outputs.hidden_states[-1].shape) # [1, 12, 4096]
with torch.no_grad():
    logits_cal = model.lm_head(seq1_outputs.hidden_states[-1])
    print(torch.abs(seq1_outputs.logits - logits_cal).sum())
# %%

# %%
# Validate seq1 and seq2 logits = seq1+seq2 logits
with torch.no_grad():
    seq2_outputs = [model(input_ids = ipt,
                        past_key_values = past_kv,
                        return_dict = True) for ipt in options_ids]
seq2_logits = [k.logits for k in seq2_outputs]
# %%
def get_logprobs(logits, targets):
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss(reduction = 'none')
    shift_logits = logits.view(-1, logits.shape[-1])
    shift_targets = targets.view(-1)
    # Enable model parallelism
    shift_targets = shift_targets.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_targets)
    return -loss
# %%
opt_logprob = [get_logprobs(
    torch.concat([seq1_outputs.logits[...,-1:,:], logits[...,:-1,:]], dim = -2), 
    ipt) 
               for ipt, logits in zip(options_ids, seq2_logits)]
# %%
opt_prob = [torch.exp(logp[:-1].sum()) for logp in opt_logprob] # exclude the eos token
print(opt_prob)
# %%
# input the whole sequence
whole_inputs = [prompt + ' ' + opt for opt in options]
whole_input_ids = [tokenizer(k, return_tensors='pt').input_ids.cuda() for k in whole_inputs]
# %%
with torch.no_grad():
    whole_outputs = [model(input_ids = ipt, return_dict = True) for ipt in whole_input_ids]
# %%
whole_logits = [k.logits for k in whole_outputs]
ipt_len = input_ids.shape[-1]
# %%
whole_logprob = [get_logprobs(logits[..., ipt_len-1:-1,:], ipt[...,ipt_len:]) 
                 for ipt, logits in zip(whole_input_ids, whole_logits)]
# whole_opt_prob = [torch.exp(logp[:-1].sum()) for logp in whole_logprob]
whole_opt_prob = [torch.exp(logp.sum()) for logp in whole_logprob]
print(whole_opt_prob)
# %%
