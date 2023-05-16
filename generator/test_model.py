from transformers import pipeline
from datasets import load_dataset
import document_retrieval
from unidecode import unidecode
import string
import random

# model_path = "generator/models/distilbert-base-1"
# 0.075

# model_path = "generator/models/distilbert-base-downstream-1"
# 0.073

# model_path = "generator/models/deberta-base-downstream-1"
# .001, dosent work for some reason

# model_path = "generator/models/deberta-base-1"
# .1

# model_path = "generator/models/deberta-base-3"
# .085

squad = load_dataset("squad", split="validation")

# shuffle indicies of squad
indicies = list(range(len(squad)))
random.seed(0)
random.shuffle(indicies)
indicies = indicies[:1000]

def get_context(question):
    hits = document_retrieval.search(question, 3)
    context = " ".join(hits)
    return context


def rough_compare(guess, page):
    """
    See if a guess is correct.  Not perfect, but better than direct string
    comparison.  Allows for slight variation.
    """
    # TODO: Also add the original answer line
    if page is None:
        return False
    
    guess = normalize_answer(guess)
    page = normalize_answer(page)

    if guess == '':
        return False
    
    if guess == page:
        return True
    elif page.find(guess) >= 0 and (len(page) - len(guess)) / len(page) > 0.5:
        return True
    else:
        return False


def normalize_answer(answer):
    """
    Remove superflous components to create a normalized form of an answer that
    can be more easily compared.
    """
    if answer is None:
        return ''
    reduced = unidecode(answer)
    reduced = reduced.replace("_", " ")
    if "(" in reduced:
        reduced = reduced.split("(")[0]
    reduced = "".join(x for x in reduced.lower() if x not in string.punctuation)
    reduced = reduced.strip()

    for bad_start in ["the ", "a ", "an "]:
        if reduced.startswith(bad_start):
            reduced = reduced[len(bad_start):]
    return reduced.strip()

question_answerer = pipeline("question-answering", model=model_path)
correct = 0
for i, idx in enumerate(indicies):
    question = squad[idx]["question"]
    context = get_context(question)
    answer = squad[idx]["answers"]["text"][0]
    
    guess = question_answerer(question=question, context=context)
    guess = guess["answer"]
    
    if rough_compare(guess, answer):
        correct += 1
    
    if i % 100 == 0:
        print(i / len(indicies))

print(correct / len(indicies))