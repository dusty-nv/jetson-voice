#!/usr/bin/env python3
# coding: utf-8

import os
import pprint

from jetson_voice.utils import load_resource


def IntentSlot(resource, *args, **kwargs):
    """
    Loads a NLP joint intent/slot classifier service or model.
    See the IntentSlotService class for the signature that implementations use.
    """
    factory_map = {
        'tensorrt' : 'jetson_voice.models.nlp.IntentSlotEngine',
        'onnxruntime' : 'jetson_voice.models.nlp.IntentSlotEngine'
    }
    
    return load_resource(resource, factory_map, *args, **kwargs)

    
class IntentSlotService():
    """
    Intent/slot classifier service base class.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Create service instance.
        """
        self.config = config
        
    def __call__(self, query):
        """
        Perform intent/slot classification on the input query.
        
        Parameters:
          query (string) -- The text query, for example:
                             'What is the weather in San Francisco tomorrow?'

        Returns a dict with the following keys:
             'intent' (string) -- the classified intent label
             'score' (float) -- the intent probability [0,1]
             'slots' (list[dict]) -- a list of dicts, where each dict has the following keys:
                  'slot' (string) -- the slot label
                  'text' (string) -- the slot text from the query
                  'score' (float) -- the slot probability [0,1]
        """
        pass
 
 
def QuestionAnswer(resource, *args, **kwargs):
    """
    Loads a NLP question answering service or model.
    See the QuestionAnswerService class for the signature that implementations use.
    """
    factory_map = {
        'tensorrt' : 'jetson_voice.models.nlp.QuestionAnswerEngine',
        'onnxruntime' : 'jetson_voice.models.nlp.QuestionAnswerEngine'
    }
    
    return load_resource(resource, factory_map, *args, **kwargs) 
        
   
class QuestionAnswerService():
    """
    Question answering service base class.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Create service instance.
        """
        self.config = config
        
    def __call__(self, query, top_k=1):
        """
        Perform question/answering on the input query.
        
        Parameters:
          query (dict or tuple) -- Either a dict with 'question' and 'context' keys,
                                   or a (question, context) tuple.
          top_k (int) -- How many of the top results to return, sorted by score.
                         The default (topk=1) is to return just the top result.
                         If topk > 1, then a list of results will be returned.
          
        Returns:
          dict(s) with the following keys:
          
             'answer' (string) -- the answer text
             'score' (float) -- the probability [0,1]
             'start' (int) -- the starting character index of the answer into the context text
             'end' (int) -- the ending character index of the answer into the context text
             
          If top_k > 1, a list of dicts with the topk results will be returned.
          If top_k == 1, just the single dict with the top score will be returned.
        """
        pass
        

def TextClassification(resource, *args, **kwargs):
    """
    Loads a NLP text classification service or model.
    See the TextClassificationService class for the signature that implementations use.
    """
    factory_map = {
        'tensorrt' : 'jetson_voice.models.nlp.TextClassificationEngine',
        'onnxruntime' : 'jetson_voice.models.nlp.TextClassificationEngine'
    }
    
    return load_resource(resource, factory_map, *args, **kwargs) 
        
   
class TextClassificationService():
    """
    Text classification service base class.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Create service instance.
        """
        self.config = config
        
    def __call__(self, query):
        """
        Perform text classification on the input query.
        
        Parameters:
          query (string) -- The text query, for example:
                             'Today was warm, sunny and beautiful out.'

        Returns a dict with the following keys:
             'class' (int) -- the predicted class index
             'label' (string) -- the predicted class label (and if there aren't labels `str(class)`)
             'score' (float) -- the classification probability [0,1]
        """
        pass


def TokenClassification(resource, *args, **kwargs):
    """
    Loads a NLP token classification (aka Named Entity Recognition) service or model.
    See the TokenClassificationService class for the signature that implementations use.
    """
    factory_map = {
        'tensorrt' : 'jetson_voice.models.nlp.TokenClassificationEngine',
        'onnxruntime' : 'jetson_voice.models.nlp.TokenClassificationEngine'
    }
    
    return load_resource(resource, factory_map, *args, **kwargs) 
        
   
class TokenClassificationService():
    """
    Token classification (aka Named Entity Recognition) service base class.
    """
    def __init__(self, config, *args, **kwargs):
        """
        Create service instance.
        """
        self.config = config
        
    def __call__(self, query):
        """
        Perform token classification (NER) on the input query and return tagged entities.
        
        Parameters:
          query (string) -- The text query, for example:
                             "Ben is from Chicago, a city in the state of Illinois, US'

        Returns a list[dict] of tagged entities with the following dictionary keys:
             'class' (int) -- the entity class index
             'label' (string) -- the entity class label
             'score' (float) -- the classification probability [0,1]
             'text'  (string) -- the corresponding text from the input query
             'start' (int) -- the starting character index of the text
             'end'   (int) -- the ending character index of the text
        """
        pass

    @staticmethod
    def tag_string(query, tags, scores=False):
        """
        Returns a string with the tags inserted inline with the query.  For example:
        
        "Ben[B-PER] is from Chicago[B-LOC], a city in the state of Illinois[B-LOC], US[B-LOC]"
        
        Parameters:
          query  (string) -- The original query string.
          tags   (list[dict]) -- The tags predicted by the model.
          scores (bool) -- If true, the probabilities will be added inline.
                           If false (default), only the tag labels will be added.
        """
        char_offset = 0

        for tag in tags:
            if scores:
                tag_str = f"[{tag['label']} {tag['score']:.3}]"
            else:
                tag_str = f"[{tag['label']}]"
                
            query = query[:tag['end'] + char_offset] + tag_str + query[tag['end'] + char_offset:]
            char_offset += len(tag_str)
            
        return query
        
        
if __name__ == "__main__":

    from jetson_voice import ConfigArgParser
    
    parser = ConfigArgParser()
    
    parser.add_argument('--model', default='distilbert_intent', type=str)
    parser.add_argument('--type', default='intent_slot', type=str)

    args = parser.parse_args()
    args.type = args.type.lower()
    
    print(args)
    
    if args.type == 'intent_slot':
    
        model = IntentSlot(args.model)
        
        # create some test queries
        queries = [
            'Set alarm for Seven Thirty AM',
            'Please increase the volume',
            'What is my schedule for tomorrow',
            'Place an order for a large pepperoni pizza from Dominos'
        ]

        # process the queries
        for query in queries:
            results = model(query)
            
            print('\n')
            print('query:', query)
            print('')
            pprint.pprint(results)
     
    elif args.type == 'question_answer' or args.type == 'qa':

        model = QuestionAnswer(args.model)
        
        # create some test queries
        queries = []
        
        queries.append({
            "question" : "What is the value of Pi?",
            "context" : "Some people have said that Pi is tasty but there should be a value for Pi, and the value for Pi is around 3.14. "
                        "Pi is the ratio of a circle's circumference to it's diameter. The constant Pi was first calculated by Archimedes "
                        "in ancient Greece around the year 250 BC."
        })
        
        queries.append({
            "question" : "Who discovered Pi?",
            "context" : queries[-1]['context']
        })

        queries.append({
            "question" : "Which nation contains the majority of the Amazon forest?",
            "context" : "The Amazon rainforest is a moist broadleaf forest that covers most of the Amazon basin of South America. "
                        "This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres "
                        "(2,100,000 sq mi) are covered by the rainforest. The majority of the forest is contained within Brazil, "
                        "with 60% of the rainforest, followed by Peru with 13%, and Colombia with 10%."
        })
        
        queries.append({
            "question" : "How large is the Amazon rainforest?",
            "context" : queries[-1]['context']
        })
        
        # process the queries
        for query in queries:
            answers = model(query, top_k=5)
            
            print('\n')
            print('context:', query['context'])
            print('')
            print('question:', query['question'])
            
            for answer in answers:
                print('')
                print('answer:  ', answer['answer'])
                print('score:   ', answer['score'])
    
    elif args.type == 'text_classification':
    
        model = TextClassification(args.model)
        
        # create some test queries (these are for sentiment models)
        queries = [
            "By the end of no such thing the audience, like beatrice, has a watchful affection for the monster.",
            "Director Rob Marshall went out gunning to make a great one.",
            "Uneasy mishmash of styles and genres.",
            "I love exotic science fiction / fantasy movies but this one was very unpleasant to watch. I gave it 4 / 10 since some special effects were nice.",
            "Today was cold and rainy and not very nice.",
            "Today was warm, sunny and beautiful out.",
        ]

        # process the queries
        for query in queries:
            results = model(query)
            print('\nquery:', query)
            pprint.pprint(results)
    
    elif args.type == 'token_classification':
    
        model = TokenClassification(args.model)
    
        # create some test queries
        queries = [
            "But candidate Charles Baker, who has about eight percent of the vote, has called for an investigation into reports of people voting multiple times.",
            "Analysts say Mr. Chung's comments may be part of efforts by South Korea to encourage North Korea to resume bilateral talks.",
            "The 63-year-old Daltrey walked offstage during the first song; guitarist Pete Townshend later told the crowd he was suffering from bronchitis and could barely speak.",
            "The Who is currently touring in support of Endless Wire, its first album since 1982.",
            "Meanwhile, Iowa is cleaning up after widespread flooding inundated homes, destroyed crops and cut off highways and bridges.",
            "At the White House Tuesday, U.S. President George Bush expressed concern for the flood victims.",
            "Ben is from Chicago, a city in the state of Illinois, US with a population of 2.7 million people.",
            "Lisa's favorite place to climb in the summer is El Capitan in Yosemite National Park in California, U.S."
        ]

        # process the queries
        for query in queries:
            tags = model(query)
            #print(f'\n{query}')
            #pprint.pprint(tags)
            print(f'\n{model.tag_string(query, tags, scores=True)}')
        
    else: 
        raise ValueError(f"invalid --type argument ({args.type})")
        