from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import document_retrieval

model_path = "qa_model/checkpoint-4000"

question_answerer = pipeline("question-answering", model=model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_answer(question, context):

    question_answerer(question=question, context=context)
    inputs = tokenizer(question, context, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens)

    return answer

def get_context(question):
    hits = document_retrieval.search(question, 3)
    context = " ".join(hits)
    return context

question = "In 1903, the first woman ever won the Nobel Prize. What was her name?"
context = get_context(question)
answer = get_answer(question, context)
print(answer)