from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

model_path = "qa_model/checkpoint-4000"

question_answerer = pipeline("question-answering", model=model_path)
question_answerer(question=question, context=context)

tokenizer = AutoTokenizer.from_pretrained(model_path)
inputs = tokenizer(question, context, return_tensors="pt")

model = AutoModelForQuestionAnswering.from_pretrained(model_path)
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
answer = tokenizer.decode(predict_answer_tokens)

print(answer)