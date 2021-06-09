#!/usr/bin/env python3
# coding: utf-8

import sys
import readline

from jetson_voice import QuestionAnswer, ConfigArgParser

parser = ConfigArgParser()
parser.add_argument('--model', default='distilbert_qa_384', type=str)
parser.add_argument('--top_k', default=1, type=int, help='show the top N answers (default 1)')
args = parser.parse_args()
print(args)

model = QuestionAnswer(args.model)  # load the QA model

builtin_context = {
    "Amazon" : "The Amazon rainforest is a moist broadleaf forest that covers most of the Amazon basin of South America. "
               "This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres "
               "(2,100,000 sq mi) are covered by the rainforest. The majority of the forest is contained within Brazil, "
               "with 60% of the rainforest, followed by Peru with 13%, and Colombia with 10%.",
    
    "Geology" : "There are three major types of rock: igneous, sedimentary, and metamorphic. Igneous rocks are formed from "
                "melted rock deep inside the Earth. Sedimentary rocks are compressed layers of sand, silt, dead plants, and "
                "animal skeletons. Metamorphic rocks are other rocks that are changed by heat and pressure underground.",
    
    "Moon Landing" : "The first manned Moon landing was Apollo 11 on July, 20 1969. The first human to step on the Moon was "
                     "astronaut Neil Armstrong followed second by Buzz Aldrin. They landed in the Sea of Tranquility with their "
                     "lunar module the Eagle. They were on the lunar surface for 2.25 hours and collected 50 pounds of moon rocks.",
           
    "Pi" : "Some people have said that Pi is tasty but there should be a value for Pi, and the value for Pi is around 3.14. "
           "Pi is the ratio of a circle's circumference to it's diameter. The constant Pi was first calculated by Archimedes "
           "in ancient Greece around the year 250 BC.",
           
    "Super Bowl 55" : "Super Bowl 55 took place on February 7, 2021 in Tampa, Florida between the Kansas City Chiefs and "
                      "the Tampa Bay Buccaneers.  The Tampa Bay Buccaneers won by a score of 31 to 9. In his first season "
                      "with Tampa Bay, it was quarterback Tom Brady's seventh Super Bowl win in nine appearances.",
}

context = builtin_context['Amazon']

def print_context():
    print('\nContext:')
    print(context)
    
def parse_commands(entry):
    """
    Parse 'C' command for changing context, 'P' to print context, and 'Q' for quit.
    Returns true if a command was entered, otherwise false.
    """
    global context

    if entry == 'C':
        print('\nSelect from one of the following topics, or enter your own context paragraph:')
        for idx, key in enumerate(builtin_context):
            print(f'   {idx+1}. {key}')
        entry = input('> ')
        try:  # try parsing as a number
            num = int(entry)
            if num > 0 and num <= len(builtin_context):
                context = builtin_context[list(builtin_context.keys())[num-1]]
            else:
                print('Invalid entry')
        except:  # try looking up topic name, otherwise custom paragraph
            if entry in builtin_context:
                context = builtin_context[entry.lower()]
            else:
                context = entry
                
        print_context()
        return True
        
    elif entry == 'P':
        print_context()
        return True
    elif entry == 'Q':
        sys.exit()
        
    return False
    
print_context()

while True:
    print('\nEnter a question, C to change context, P to print context, or Q to quit:')
    entry = input('> ')
    
    if parse_commands(entry.upper()):
        continue
    
    query = {
        'context' : context,
        'question' : entry
    }
    
    results = model(query, top_k=args.top_k)
    
    if args.top_k == 1:
        results = [results]
        
    for result in results:
        print('\nAnswer:', result['answer'])
        print('Score: ', result['score'])
        