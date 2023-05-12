
import json
from tfidf_guesser import TfidfGuesser
from unidecode import unidecode
import string

def get_guesser():
    guesser = TfidfGuesser("tfidf_model/TfidfGuesser")
    guesser.load()
    return guesser

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


def main():

    split = 0.8
    data_split = "train"
    data_path = "data/qanta.buzztrain.json"

    final_data = {}

    guesser = get_guesser()

    with open(data_path, "r") as infile:
        data = json.load(infile)

    this_data = {
        "text": [],
        "label": []
    }

    for i, question in enumerate(data):
        question_text = question["text"]
        page = question["page"]

        if i / len(data) > split:
            final_data[data_split] = this_data
            data_split = "test"
            this_data = {
                "text": [],
                "label": []
            }

        guesses = guesser(question_text)

        for guess in guesses:
            guess_text = guess["guess"]
            this_data["text"].append(question_text)
            if rough_compare(guess_text, page):
                this_data["label"].append(1)
            else:
                this_data["label"].append(0)

        if i % 100 == 0:
            print(i / len(data))

    final_data[data_split] = this_data
    
    with open("data/classification.qanta.buzztrain.json", "w") as outfile:
        json.dump(final_data, outfile)

if __name__ == "__main__":
    main()