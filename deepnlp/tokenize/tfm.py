"""
Utilities related to transformers.PreTrainedTokenizer
"""

from typing import List, Dict, Tuple, Union, Optional, Any
import itertools
from transformers import PreTrainedTokenizer

def pad_tokens(token_ids, max_length: Optional[int], pad_id = 0):
    """Pad text to a specified length. If overflow, ignore."""
    token_ids = list(token_ids)
    if max_length is None or len(token_ids) > max_length:
        paded_ids = token_ids
        mask = [1] * len(token_ids)
    else:
        difference = max_length - len(token_ids)
        paded_ids = token_ids + [pad_id] * difference
        mask = [1] * len(token_ids) + [0] * difference
    return paded_ids, mask

def truncate_add_pad(token_ids, tokenizer: PreTrainedTokenizer,
                     max_length = None):
    """truncate, add special token and pad"""
    n_spec = len(tokenizer.build_inputs_with_special_tokens([]))
    # truncate
    if max_length is not None:
        token_ids = token_ids[:max_length - n_spec]
    
    # add spectial token
    required_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
    
    # pad and mask
    paded_ids, mask = pad_tokens(required_ids, max_length, tokenizer.pad_token_id)
    
    return paded_ids, mask

# If you just want to do tokenization, use this func.
def encode_text(
    text: Union[str, List[str]],
    tokenizer: PreTrainedTokenizer,
    max_length = None
)->Tuple[List[int], List[int]]:
    """
    Convert a sentence (str or list of words) into ids and prepare for model.

    This function is eauql to:
    ```
    tokenizer(text, padding = 'max_length', truncation = True, max_length = max_length, )
    ```
    """
    if isinstance(text, str):
        tokens = tokenizer.tokenize(text)
    elif isinstance(text, (list, tuple)):
        tokens = list(itertools.chain(*[tokenizer.tokenize(t) for t in text]))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids, mask = truncate_add_pad(token_ids, tokenizer, max_length)
    
    return token_ids, mask

def truncate_add_pad_mapping(
    token_ids, tokenizer: PreTrainedTokenizer, 
    all_word_to_tokens: List[Tuple[int, int]], 
    max_length = None
)-> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """
    Truncate, add special token and pad with word mapping.
    No truncation if max_length is None.
    Cut the last word if exceeding the max_length
    """
    # n_spec = tokenizer.num_special_tokens_to_add()
    n_spec = len(tokenizer.build_inputs_with_special_tokens([]))
    # truncate
    if max_length is not None:
        token_ids = token_ids[:max_length - n_spec]
        all_word_to_tokens = [k for k in all_word_to_tokens if k[0] < max_length - n_spec]
        if len(all_word_to_tokens) > 0:
            last_span = all_word_to_tokens[-1]
            if last_span[1] > max_length - n_spec:
                all_word_to_tokens[-1] = (last_span[0], max_length - n_spec)
    
    # add spectial token and offset the mapping
    required_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
    head_pad_n = required_ids.index(token_ids[0])
    if head_pad_n > 0:
        for i, word_to_tokens in enumerate(all_word_to_tokens):
            all_word_to_tokens[i] = [k + head_pad_n for k in word_to_tokens]
    
    # pad and mask
    paded_ids, mask = pad_tokens(required_ids, max_length, tokenizer.pad_token_id)
    
    return paded_ids, mask, all_word_to_tokens

# If you want to perserve the original word boundaries, use this func.
def encode_word_list(
    word_list: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length = None
)-> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """
    Encode words and get word to token mapping.

    Args:
        word_list: assume obtained by a whitespace tokenizer.
        
    Return:
        (token_ids, mask, all_word_to_tokens)
        **token_ids**: List[int]
        **mask**: List[int]
        **all_word_to_tokens**(List[TokenSpan]):
            token span for words
            - **start** -- Index of the first token.
            - **end** -- Index of the token following the last token.
    """
    word_sub_tokens = [tokenizer.tokenize(t) for t in word_list]
    all_word_to_tokens = [] # List[Tuple[start, end]]
    all_token_to_word = [] # List[int]
    past_n_sub = 0
    for i, word in enumerate(word_sub_tokens):
        left = past_n_sub
        right = past_n_sub + len(word)
        all_word_to_tokens.append((left, right))
        all_token_to_word.extend([i] * len(word))
        past_n_sub += len(word)
    
    tokens = list(itertools.chain(*word_sub_tokens))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids, mask, all_word_to_tokens = truncate_add_pad_mapping(token_ids, tokenizer, all_word_to_tokens, max_length)

    return token_ids, mask, all_word_to_tokens