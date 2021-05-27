#!/usr/bin/env python3
# coding: utf-8

import nltk
import numpy as np

from jetson_voice.utils import global_config


def transcript_from_words(words, scores=False, times=False, end=False):
    """
    Convert a list of words to the text transcript.
    """
    transcript = ''
    
    for idx, word in enumerate(words):
    
        if scores and times:
            transcript += f"{word['text']} ({word['start_time']}:{word['end_time']} {word['score']:.2f})"
        elif scores:
            transcript += f"{word['text']} ({word['score']:.2f})"
        elif times:
            transcript += f"{word['text']} ({word['start_time']}:{word['end_time']})"
        else:
            transcript += word['text']
        
        if idx < len(words) - 1:
            transcript += ' '
      
    if end:
        transcript += '.'  # add punctuation to end
      
    return transcript
        

def find_overlapping_word(wordlist, word):
    """
    Find the first word from the list with overlapping times.
    Returns a (word, index) tuple or (None, -1) if no overlap found.
    """
    for idx, word2 in enumerate(wordlist):
        if not (word['end_time'] < word2['start_time'] or word['start_time'] > word2['end_time']):
            return word2, idx 
    return None, -1


def find_word_after(wordlist, time):
    """
    Find the nearest word that starts after the time.
    Returns a (word, index) tuple or (None, 1) if all words start before the time.
    """
    if isinstance(time, tuple):
        time = time[1]  # use the end time
        
    for idx, word in enumerate(wordlist):
        if time <= word['start_time']:
            return word, idx        
            
    return None, -1


def find_word_before(wordlist, time):
    """
    Find the nearest word that starts after the time.
    Returns a (word, index) tuple or (None, 1) if all words start after the time.
    """
    if isinstance(time, tuple):
        time = time[0]  # use the start time
        
    for idx in range(len(wordlist)-1, -1, -1):
        if time >= wordlist[idx]['end_time']:
            return wordlist[idx], idx    
            
    return None, -1


def merge_words(wordlist, words, score_threshold=-np.inf, method='overlap'):
    """
    Merge new words with past words.  This works by finding overlapping or similar words,
    and replacing the old word with new word if the new word has a higher probability.
    """
    if len(words) == 0:
        return wordlist
        
    if len(wordlist) == 0:
        return words
        
    # short-circuit if these are all new words    
    if words[0]['start_time'] > wordlist[-1]['end_time']:
        wordlist.extend(words)
        return wordlist
         
    if method == 'overlap':
        # find words that overlap and pick the highest-scoring one
        for word in words:
            if word['score'] < score_threshold: #self.config['word_threshold']:
                continue
                
            if len(wordlist) == 0 or word['start_time'] > wordlist[-1]['end_time']:
                wordlist.append(word)
                continue

            overlap_word, overlap_idx = find_overlapping_word(wordlist, word)
            
            if overlap_word is None:
                continue

            if global_config.debug:
                print(f"found new '{word['text']}' ({word['start_time']}:{word['end_time']} {word['score']:.2f}) overlaps with '{overlap_word['text']}' ({overlap_word['start_time']}:{overlap_word['end_time']} {overlap_word['score']:.2f})")

            if word['score'] > overlap_word['score']:
                wordlist[overlap_idx] = word
                
    elif method == 'similarity':
        # find the most-similar past word to the first new word
        similarity_metric = np.inf #1000
        similarity_index = -1
        
        for idx in range(len(wordlist)-1, -1, -1):  # search in reverse so words early in the transcript aren't matched first
            similarity = nltk.edit_distance(words[0]['text'], wordlist[idx]['text'])
            
            if similarity < similarity_metric:
                similarity_metric = similarity
                similarity_index = idx
                
            if similarity == 0:
                break
           
        if global_config.debug:
            print(f"closest word to '{words[0]['text']}' is '{wordlist[similarity_index]['text']}' (similarity={similarity_metric}) ")
        
        wordlist = wordlist[:similarity_index]
        wordlist.extend(words)
        
    else:
        raise ValueError(f"invalid method '{method}' (valid options are 'overlap', 'similarity')")
        
    return wordlist
        
        
def split_words(wordlist, times):
    """
    Split the word list by the given times.
    note - these times should be sorted
    """
    wordlists = []

    for time in times:
        _, idx = find_word_after(wordlist, time)
        
        if idx < 0:
            wordlists.append(wordlist)
            return wordlists
            
        wordlists.append(wordlist[:idx])
        wordlist = wordlist[idx:]
        
    wordlists.append(wordlist)    
    return wordlists
        
        
def rebase_word_times(wordlist):
    """
    Re-base the word timings so that the start of the first word is zero.
    """
    if len(wordlist) == 0:
        return wordlist
        
    #wordlist = wordlist.copy()
    start_offset = wordlist[0]['start_time']
            
    for idx in range(len(wordlist)):
        wordlist[idx]['start_time'] -= start_offset
        wordlist[idx]['end_time'] -= start_offset
    
    return wordlist


def find_silent_intervals(logits, blank_symbol_id, min_silent_time, time_offset):
    """
    Find blank/silent regions in the output logits.
    """
    num_timesteps = logits.shape[0]
    silent_intervals = []
    last_interval_start = None
    
    for i in range(num_timesteps):
        argmax = np.argmax(logits[i])
        
        if argmax == blank_symbol_id:
            if last_interval_start is None:
                last_interval_start = i 
        
        if last_interval_start is not None and (argmax != blank_symbol_id or (i == num_timesteps-1)):
            if i - last_interval_start >= min_silent_time:
                silent_intervals.append((last_interval_start + time_offset, i-1+time_offset))
            #    print(f'     new silent interval ({last_interval_start + self.timestep}:{i-1+self.timestep}) {i - last_interval_start} > {min_length:.2f}')  
            #else:
            #    print(f'skipping silent interval ({last_interval_start + self.timestep}:{i-1+self.timestep}) {i - last_interval_start} < {min_length:.2f}')
                
            last_interval_start = None

    return silent_intervals
        
