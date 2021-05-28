#!/usr/bin/env python3
# coding: utf-8

import os
import logging
import numpy as np

from transformers import AutoTokenizer

from jetson_voice.nlp import QuestionAnswerService
from jetson_voice.utils import load_model, normalize_logits
from .nlp_utils import nlp_dynamic_shapes


class QuestionAnswerEngine(QuestionAnswerService):
    """
    Question answering model in TensorRT / onnxruntime.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Load an question answering model from ONNX
        """
        super(QuestionAnswerEngine, self).__init__(config, *args, **kwargs)

        if self.config.type != 'qa':
            raise ValueError(f"{self.config.model_path} isn't a Question Answering model (type '{self.config.type}'")
            
        # load model
        dynamic_shapes = {'max' : (1, self.config['dataset']['max_seq_length'])}  # (batch_size, sequence_length)
        
        if nlp_dynamic_shapes:
            dynamic_shapes['min'] = (1, 1)
        
        self.model = load_model(self.config, dynamic_shapes)
        
        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer']['tokenizer_name'])
        self.question_first = bool(self.tokenizer.padding_side == "right")
        
        
    def __call__(self, query, top_k=1):
        """
        Perform question/answering on the input query.
        
        Parameters:
          query (dict or tuple) -- Either a dict with 'question' and 'context' keys,
                                   or a (question, context) tuple.
          top_k (int) -- How many of the top results to return, sorted by score.
                         The default (top_k=1) is to return just the top result.
                         If top_k > 1, then a list of results will be returned.
          
        Returns:
          dict(s) with the following keys:
          
             'answer' (string) -- the answer text
             'score' (float) -- the probability [0,1]
             'start' (int) -- the starting character index of the answer into the context text
             'end' (int) -- the ending character index of the answer into the context text
             
          If top_k > 1, a list of dicts with the top_k results will be returned.
          If top_k == 1, just the single dict with the top score will be returned.
        """
        if isinstance(query, dict):
            question = query['question']
            context = query['context']
        elif isinstance(query, tuple):
            question = query[0]
            context = query[1]
        else:
            raise ValueError(f'query must be a dict or tuple (instead was type {type(query).__name__})')

        # check for models that have a doc_stride >= max_seq_length
        # this will cause an exception in the tokenizer
        doc_stride = self.config['dataset']['doc_stride']
        max_seq_len = self.config['dataset']['max_seq_length']
        
        if doc_stride >= max_seq_len:
            doc_stride = int(max_seq_len/2)
            
        # tokenize the inputs
        encodings = self.tokenizer(
            text=question if self.question_first else context,
            text_pair=context if self.question_first else question_text,
            padding='longest' if nlp_dynamic_shapes else 'max_length',
            truncation="only_second" if self.question_first else "only_first",
            max_length=max_seq_len,
            stride=doc_stride,
            return_tensors='np',
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        
        # When the input is too long, it's converted in a batch of inputs with overflowing tokens
        # and a stride of overlap between the inputs. If a batch of inputs is given, a special output
        # "overflow_to_sample_mapping" indicate which member of the encoded batch belong to which original batch sample.
        # Here we tokenize examples one-by-one so we don't need to use "overflow_to_sample_mapping".
        # "num_span" is the number of output samples generated from the overflowing tokens.
        num_spans = len(encodings["input_ids"])
        logging.debug(f'num_spans: {num_spans}')

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # We put 0 on the tokens from the context and 1 everywhere else (question and special tokens)
        p_mask = np.asarray(
            [
                [tok != 1 if self.question_first else 0 for tok in encodings.sequence_ids(span_id)]
                for span_id in range(num_spans)
            ]
        )

        # keep the cls_token unmasked (some models use it to indicate unanswerable questions)
        if self.tokenizer.cls_token_id is not None:
            cls_index = np.nonzero(encodings["input_ids"] == self.tokenizer.cls_token_id)
            p_mask[cls_index] = 0
            
        # run the model over each span (TODO batching)
        model_outputs = []
        
        for span_idx in range(num_spans):
            inputs = {}
            
            for input in self.model.inputs:
                if input.name not in encodings:
                    raise ValueError(f"the encoded inputs from the tokenizer doesn't contain '{input.name}'")

                inputs[input.name] = np.expand_dims(encodings[input.name][span_idx], axis=0) # add batch dim

            model_outputs.append(self.model.execute(inputs))
            
        # post-processing
        answers = []
        min_null_score = 1000000
        handle_impossible_answer = self.config['dataset']['version_2_with_negative']
        
        for span_idx in range(num_spans):
            start_logits = np.squeeze(model_outputs[span_idx][:,:,0])
            end_logits = np.squeeze(model_outputs[span_idx][:,:,1])

            # Ensure padded tokens & question tokens cannot belong to the set of candidate answers.
            undesired_tokens = np.abs(p_mask[span_idx] - 1) & encodings['attention_mask'][span_idx]

            # Generate mask
            undesired_tokens_mask = (undesired_tokens == 0.0)

            # Make sure non-context indexes in the tensor cannot contribute to the softmax
            start_logits = np.where(undesired_tokens_mask, -10000.0, start_logits)
            end_logits = np.where(undesired_tokens_mask, -10000.0, end_logits)

            # Normalize logits and spans to retrieve the answer
            start_logits = np.exp(start_logits - np.log(np.sum(np.exp(start_logits), axis=-1, keepdims=True)))
            end_logits = np.exp(end_logits - np.log(np.sum(np.exp(end_logits), axis=-1, keepdims=True)))

            if handle_impossible_answer:
                min_null_score = min(min_null_score, (start_logits[0] * end_logits[0]).item())

            # Mask CLS
            start_logits[0] = end_logits[0] = 0.0

            # Decode token probabilities
            starts, ends, scores = self.decode(start_logits, end_logits, top_k=top_k)

            if self.tokenizer.is_fast:
                # Convert the answer (tokens) back to the original text
                # Score: score from the model
                # Start: Index of the first character of the answer in the context string
                # End: Index of the character following the last character of the answer in the context string
                # Answer: Plain text of the answer
                enc = encodings[span_idx]
                
                # Sometimes the max probability token is in the middle of a word so:
                # - we start by finding the right word containing the token with `token_to_word`
                # - then we convert this word in a character span with `word_to_chars`
                for s, e, score in zip(starts, ends, scores):
                    start = enc.word_to_chars(enc.token_to_word(s), sequence_index=1 if self.question_first else 0)[0]
                    end = enc.word_to_chars(enc.token_to_word(e), sequence_index=1 if self.question_first else 0)[1]
                    
                    answers.append({
                        'answer' : context[start : end],
                        'score' : score.item(),
                        'start' : start,
                        'end' : end
                    })
            else:
                raise NotImplementedError('QA post-processing is only implemented for fast tokenizers')
            
        if handle_impossible_answer:
            answers.append({'answer': '', 'score': min_null_score, 'start': 0, 'end': 0})

        answers = sorted(answers, key=lambda x: x['score'], reverse=True)[:top_k]
        
        if top_k == 1:
            return answers[0]
        else:
            return answers


    def decode(self, start: np.ndarray, end: np.ndarray, top_k: int):
        """
        Take the QA model output and will generate probabilities for each span to be the actual answer.
        In addition, it filters out some unwanted/impossible cases like answer len being greater than max_answer_len or
        answer end position being before the starting position. The method supports output the k-best answer through
        the top_k argument.
        Args:
            start (:obj:`np.ndarray`): Individual start probabilities for each token.
            end (:obj:`np.ndarray`): Individual end probabilities for each token.
            top_k (:obj:`int`): Indicates how many possible answer span(s) to extract from the model output.
            max_answer_len (:obj:`int`): Maximum size of the answer to extract from the model's output.
        """
        # Ensure we have batch axis
        if start.ndim == 1:
            start = start[None]

        if end.ndim == 1:
            end = end[None]

        # Compute the score of each tuple(start, end) to be the real answer
        outer = np.matmul(np.expand_dims(start, -1), np.expand_dims(end, 1))

        # Remove candidate with end < start and end - start > max_answer_len
        candidates = np.tril(np.triu(outer), self.config['dataset']['max_answer_length'] - 1)

        #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
        scores_flat = candidates.flatten()
        if top_k == 1:
            idx_sort = [np.argmax(scores_flat)]
        elif len(scores_flat) < top_k:
            idx_sort = np.argsort(-scores_flat)
        else:
            idx = np.argpartition(-scores_flat, top_k)[0:top_k]
            idx_sort = idx[np.argsort(-scores_flat[idx])]

        start, end = np.unravel_index(idx_sort, candidates.shape)[1:]
        return start, end, candidates[0, start, end] 
        