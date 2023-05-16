from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_from_disk, load_dataset


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                                          
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["new_context"],
        max_length=384,
        truncation=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        print(offset)

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

squad = load_from_disk("generator/data/squad_bad_test")
squad = squad.train_test_split(test_size=0.2)

tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

print(tokenized_squad['train'][0])

answer_in_context = 0
for i, example in enumerate(tokenized_squad['train']):
    if example["start_positions"] != 0 or example["end_positions"] != 0:
        answer_in_context += 1
    
    if i % 10000 == 0:
        print(i / len(tokenized_squad['train']))

print()
print(answer_in_context / len(tokenized_squad['train']))