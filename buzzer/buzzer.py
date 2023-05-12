from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="buzzer_model")

def get_output(text):
    output = classifier(text)
    return output