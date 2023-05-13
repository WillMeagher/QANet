import bert_guess
import spacy
import qbmodel
# model = bert_guess.BertGuess()
# nlp = spacy.load('en_core_web_lg')
# questions = 'data/small.guessdev.json'
# questions = model.json_to_dict(questions)
# total_loss = 0
# for question in questions[:10]:
#     doc1 = model(question['text'], 5, 'chunk')[0]
#     doc2 = question['answer']
#     print("--------------------------------------------------")
#     print("Question:")
#     print(question['text'])
#     print("Model Prediction:")
#     print(doc1)
#     print("Actual Answer:")
#     print(doc2)
#     print("Similarity:")
#     similarity = nlp(doc1['answer']).similarity(nlp(doc2))
#     print(similarity)
#     total_loss += (1 - similarity)

# average_loss = total_loss / len(questions)
# print("--------------------------------------------------")
# print("Average loss per question:")
# print(average_loss)

model = qbmodel.QuizBowlModel()
question = "This poet noted that he \"had no human fears\" in a quintet of poems that centered on \"A Maid whom there were none to praise / And very few to love.\" In another poem, he called his home country \"a fen / Of stagnant waters\" and wished that \"Milton!\" \"shouldst be living at this hour.\" In one poem, he described objects which \"flash upon that inward eye\" and provide \"the bliss of solitude.\" In another poem, he wrote that he'd \"rather be / A Pagan suckled in a creed outworn.\" This author included \"She dwelt among the untrodden ways\" and \"A slumber did my spirit seal\" in his Lucy poems. For 10 points, name this poet of \"London, 1802\" and \"The World is Too Much with Us,\" who described seeing \"a host of golden daffodils\" in \"I Wandered Lonely as a Cloud\" and wrote \"Tintern Abbey."
ans = "William Wordsworth"
guess = model.guess_and_buzz([question])[0]
print(guess)