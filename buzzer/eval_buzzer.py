# evaluate the model both the buzzer and guesser

import random
from train_buzzer import rough_compare
import logging
from tqdm import tqdm

kLABELS = {"best": "Guess was correct, Buzz was correct",
           "timid": "Guess was correct, Buzz was not",
           "hit": "Guesser ranked right page first",
           "close": "Guesser had correct answer in top n list",
           "miss": "Guesser did not have correct answer in top n list",
           "aggressive": "Guess was wrong, Buzz was wrong",
           "waiting": "Guess was wrong, Buzz was correct"}

def eval_retrieval(guesser, questions, n_guesses = 1, cutoff=-1):
    """
    Evaluate the retrieval model
    """
    from collections import defaultdict, Counter
    outcomes = Counter()
    examples = defaultdict(list)
    question_text = []


    for question in tqdm(questions):
        text = question["text"]
        if cutoff == 0:
            text = text[:int(random.random() * len(text))]
        elif cutoff > 0:
            text = text[:cutoff]
        question_text.append(text)

    all_guesses = guesser.batch_guess(question_text, n_guesses)
    assert len(all_guesses) == len(question_text)
    for question, guesses, text in zip(questions, all_guesses, question_text):
        if len(guesses) > n_guesses:
            logging.warn("Warning: guesser is not obeying n_guesses argument")
            guesses = guesses[:n_guesses]
            
        top_guess = guesses[0]["guess"]
        answer = question["page"]

        example = {"text": text, "guess": top_guess, "answer": answer, "id": question["qanta_id"]}

        if any(rough_compare(x["guess"], answer) for x in guesses):
            outcomes["close"] += 1
            if rough_compare(top_guess, answer):
                outcomes["hit"] += 1
                examples["hit"].append(example)
            else:
                examples["close"].append(example)
        else:
            outcomes["miss"] += 1
            examples["miss"].append(example)

    return outcomes, examples

def eval_buzzer(buzzer, questions):
    """
    Compute buzzer outcomes on a dataset
    """
    
    from collections import Counter, defaultdict
    
    buzzer.load()
    buzzer.add_data(questions)
    buzzer.build_features()
    
    predict, feature_matrix, feature_dict, correct, metadata = buzzer.predict(questions)
    outcomes = Counter()
    examples = defaultdict(list)
    for buzz, guess_correct, features, meta in zip(predict, correct, feature_dict, metadata):
        # Add back in metadata now that we have prevented cheating in feature creation
        for ii in meta:
            features[ii] = meta[ii]
        if guess_correct:
            if buzz:
                outcomes["best"] += 1
                examples["best"].append(features)
            else:
                outcomes["timid"] += 1
                examples["timid"].append(features)
        else:
            if buzz:
                outcomes["aggressive"] += 1
                examples["aggressive"].append(features)
            else:
                outcomes["waiting"] += 1
                examples["waiting"].append(features)
    return outcomes, examples

def pretty_feature_print(features, first_features=["guess", "answer", "id"]):
    """
    Nicely print a buzzer example's features
    """
    
    import textwrap
    wrapper = textwrap.TextWrapper()

    lines = []

    for ii in first_features:
        lines.append("%20s: %s" % (ii, features[ii]))
    for ii in [x for x in features if x not in first_features]:
        if isinstance(features[ii], str):
            if len(features[ii]) > 70:
                long_line = "%20s: %s" % (ii, "\n                      ".join(wrapper.wrap(features[ii])))
                lines.append(long_line)
            else:
                lines.append("%20s: %s" % (ii, features[ii]))
        elif isinstance(features[ii], float):
            lines.append("%20s: %0.4f" % (ii, features[ii]))
        else:
            lines.append("%20s: %s" % (ii, str(features[ii])))
    lines.append("--------------------")
    return "\n".join(lines)


if __name__ == "__main__":
    # Load model and evaluate it
    import argparse
    import json

    filePath = "../data/qanta.train.json"
    with open(filePath, "r") as infile:
        questions = json.load(infile)["questions"]

    buzzer = # im not sure how to load the buzzer model since i probabvly dont want to use train_buzzer every time i want to evaluate it
    outcomes, examples = eval_buzzer(buzzer, questions)


    total = sum(outcomes[x] for x in outcomes if x != "hit")
    for ii in outcomes:
        print("%s %0.2f\n===================\n" % (ii, outcomes[ii] / total))
        if len(examples[ii]) > 10:
            population = list(random.sample(examples[ii], 10))
        else:
            population = examples[ii]
        for jj in population:
            print(pretty_feature_print(jj))
        print("=================")
        
    for weight, feature in zip(buzzer._classifier.coef_[0], buzzer._featurizer.feature_names_):
        print("%40s: %0.4f" % (feature.strip(), weight))
    print("Questions Right: %i (out of %i) Accuracy: %0.2f  Buzz ratio: %0.2f" %
          (outcomes["best"], total, (outcomes["best"] + outcomes["waiting"]) / total,
           outcomes["best"] - outcomes["aggressive"] * 0.5))
