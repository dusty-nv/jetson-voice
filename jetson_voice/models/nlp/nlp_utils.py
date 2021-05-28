#!/usr/bin/env python3
# coding: utf-8

import numpy as np


# NLP BERT models (and BERT derivatives) have myelin problem with dynamic shapes on aarch64,
# so we disable dynamic shape changing for now (shapes will be set to the max sequence length)
nlp_dynamic_shapes=False


def find_subtokens(encodings, method='char_span'):
    """
    Compute the subtoken mask, where each token is marked as True if it's a subtoken or False otherwise.
    Longer words/acronyms may be tokenized into mulitple word pieces (called subtokens), for example:
    
        'Yosemite' -> ['yo', '##se', '##mite']
        'U.S.' -> ['u', '.', 's', '.']
    
    Parameters:
      encodings (BatchEncoding) -- Output from tokenizer
      
      method (string) -- If 'char_span', the subtoken mask will be determined by looking at the character
                         indices.  Tokens that map to characters that are side-by-side are flagged as subtokens.
                         
                         If 'subtoken_delimiters', subtokens will be identified by looking for '##' symbols.
                         However this can miss punctuated subtokens, such as 'U.S.'
    
    Returns boolean subtoken mask array with shape (num_queries, num_tokens)
    """
    num_queries = encodings['input_ids'].shape[0]
    subtoken_mask = []
    
    if method == 'char_span':
        for query_idx in range(num_queries):
            mask = []
            last_char = -1
            tokens = encodings.tokens(query_idx)
            
            for token_idx, word_id in enumerate(encodings.word_ids(query_idx)):
                if word_id is None:  # skip special tokens
                    mask.append(False)
                    continue
                    
                chars = encodings.token_to_chars(query_idx, token_idx)
                
                if chars[0] == last_char:
                    mask.append(True)
                else:
                    mask.append(False)
                    
                last_char = chars[1]

            subtoken_mask.append(mask)
            
    elif method == 'subtoken_delimiters':
        for query_idx in range(num_queries):
            subtoken_mask.append([token.startswith('##') for token in encodings.tokens(query_idx)])
    else:
        raise ValueError(f"invalid method ('{method}')")
        
    return np.asarray(subtoken_mask)
        