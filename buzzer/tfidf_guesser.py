from collections import defaultdict
import pickle

import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = 'tfidf.pickle'
INDEX_PATH = 'index.pickle'
QN_PATH = 'questions.pickle'
ANS_PATH = 'answers.pickle'

from guesser import print_guess, Guesser


class MyVectorizer:
    def __init__(self, width):
        self.width = width
        self.vectorizer = None

    def fit_transform(self, questions):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=self.width, stop_words='english', ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
        self.vectorizer.fit(questions)
    
    def transform(self, questions):
        return self.vectorizer.transform(questions)

class TfidfGuesser(Guesser):
    """
    Class that, given a query, finds the most similar question to it.
    """
    def __init__(self, filename, width=1000000):
        """
        Initializes data structures that will be useful later.
        filename -- base of filename we store vectorizer and documents to
        min_df -- we use the sklearn vectorizer parameters, this for min doc freq
        max_df -- we use the sklearn vectorizer parameters, this for max doc freq
        """

        # You'll need add the vectorizer here and replace this fake vectorizer
        self.tfidf_vectorizer = MyVectorizer(width)
        self.tfidf = None 
        self.questions = None
        self.answers = None
        self.filename = filename

    def __call__(self, question, max_n_guesses=4):
        """
        Given the text of questions, generate guesses (a list of both both the page id and score) for each one.
        Keyword arguments:
        question -- Raw text of the question
        max_n_guesses -- How many top guesses to return
        """
        guesses = []

        # Compute the cosine similarity
        question_tfidf = self.tfidf_vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_tfidf, self.tfidf)
        cos = cosine_similarities[0]
        indices = cos.argsort()[::-1]

        answer_values = defaultdict(list)
        answer_questions = defaultdict(list)

        for i in range(min(len(indices), max_n_guesses * 5)):
            idx = indices[i]
            answer = self.answers[idx]

            answer_values[answer].append(cos[idx])
            if cos[idx] == max(answer_values[answer]):
                answer_questions[answer] = self.questions[idx]

        for answer in answer_values:
            answer_values[answer] = math.sqrt(sum(x**2 for x in answer_values[answer]))

        sorted_answers = sorted(answer_values, key=answer_values.get, reverse=True)

        for i in range(min(max_n_guesses, len(sorted_answers))):
            # The line below is wrong but lets the code run for the homework.
            # Remove it or fix it!
            answer = sorted_answers[i]
            question = answer_questions[answer]
            confidence = answer_values[answer]
            guess =  {"question": question, "guess": answer, "confidence": confidence}
            guesses.append(guess)
        return guesses
    
    def load(self):
        """
        Load the tf-idf guesser from a file
        """
        
        path = self.filename
        with open("%s.vectorizer.pkl" % path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        with open("%s.tfidf.pkl" % path, 'rb') as f:
            self.tfidf = pickle.load(f)
        
        with open("%s.questions.pkl" % path, 'rb') as f:
            self.questions = pickle.load(f)

        with open("%s.answers.pkl" % path, 'rb') as f:
            self.answers = pickle.load(f)


if __name__ == "__main__":
    text = "What is the capital of France?"

    guesser = TfidfGuesser("tfidf_model/TfidfGuesser")
    guesser.load()

    guesses = guesser(text)

    for guess in guesses:
        print(print_guess(guess, 100))
    
