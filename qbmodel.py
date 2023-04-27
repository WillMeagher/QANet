from typing import List, Tuple
import torch as th
import nltk
import sklearn
import numpy as np
import pandas as pd

"""
Our plan is to use a RNN to predict the answers to questions.
 We plan to either use the LSTM model or some common variation on 
 it. In addition, we hope to add inputs to the model that are the 
 results of a search engine query about parts of the question. 
 This means that after a question is input the model will perform 
 some searches to gain more information about the topics in the 
 question and hopefully come to a more informed answer. 

Some techniques we are considering attempting to implement 
regarding using a search engine include using top results as 
a bag of words feature and inputting top results word by word 
into the RNN.

Other techniques that we may use include word embeddings such as 
word2vec, part-of-speech tagging, and tf-idf when creating 
features/inputs to our RNN. We will also be required to use 
the techniques associated with training a network such as an RNN. 

We will look into having different output types from our RNN 
such as a vector that represents Wikipedia pages, letting 
the model output word embeddings, and having an output that 
represents confidence which will allow the model to decide 
when to buzz. 


What needs to be done:
- First lets make the model work with the whole question, later we can implement runs.
- Quanta can be split into two parts, the guessor and buzzer.
- The guessor generates the guess and the buzzer decides when to buzz.
- We should also have a evaluator, but that can be done later.

"""
class QuizBowlModel:

    def __init__(self):
        """
        Load your model(s) and whatever else you need in this function.

        Do NOT load your model or resources in the guess_and_buzz() function, 
        as it will increase latency severely. 
        """
        pass

    def guess_and_buzz(self, question_text: List[str]) -> List[Tuple[str, bool]]:
        """
        This function accepts a list of question strings, and returns a list of tuples containing
        strings representing the guess and corresponding booleans representing 
        whether or not to buzz. 

        So, guess_and_buzz(["This is a question"]) should return [("answer", False)]

        If you are using a deep learning model, try to use batched prediction instead of 
        iterating using a for loop.
        """
        pass
