from transformers import pipeline
import document_retrieval

model_path = "generator/models/deberta-base-downstream-1"
# model_path = "generator/models/deberta-base-3"

question_answerer = pipeline("question-answering", model=model_path)

def get_answer(question, context):
    answer = question_answerer(question=question, context=context)    
    return answer

def get_context(question):
    hits = document_retrieval.search(question, 3)
    context = " ".join(hits)
    return context

question = "How tall is the Eiffel tower?"

context = get_context(question)
print(context)

answer = get_answer(question, context)
print(answer["answer"])