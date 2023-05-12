import bert_guess
import spacy
model = bert_guess.BertGuess()
nlp = spacy.load('en_core_web_sm')
questions = 'data/small.guessdev.json'
questions = model.json_to_dict(questions)
total_loss = 0
for question in questions:
    doc1 = model(question['text'], 5, 'simple')[0]
    doc2 = question['answer']
    print("--------------------------------------------------")
    print("Question:")
    print(question['text'])
    print("Model Prediction:")
    print(doc1)
    print("Actual Answer:")
    print(doc2)
    print("Similarity:")
    similarity = nlp(doc1[0]['answer']).similarity(nlp(doc2[0]['answer']))
    print(similarity)
    total_loss += (1 - similarity)

average_loss = total_loss / len(questions)
print("--------------------------------------------------")
print("Average loss per question:")
print(average_loss)