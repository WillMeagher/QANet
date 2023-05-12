from tfidf_guesser import TfidfGuesser
import buzzer

question = "What is the capital of France?"

def get_guesser():
    guesser = TfidfGuesser("tfidf_model/TfidfGuesser")
    guesser.load()
    return guesser

def print_guesses(guesses):
    for guess in guesses:
        print(guess)

guesser = get_guesser()
guesses = guesser(question, max_n_guesses=10)
# print_guesses(guesses)

max_confidence = 0
found_answer = False
actual_guess = None

for guess in guesses:
    buzzer_input = question + " " + guess["guess"]
    output = buzzer.get_output(buzzer_input)

    if output[0]["label"] == "RIGHT" and output[0]["score"] > max_confidence:
        found_answer = True
        max_confidence = output[0]["score"]
        actual_guess = guess

if found_answer:
    print("Found answer: " + actual_guess["guess"])
    
    