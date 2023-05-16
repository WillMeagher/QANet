import document_retrieval
from datasets import load_dataset

squad = load_dataset("squad", split="train")

new_contexts = []
for i in range(len(squad)):
    # new_context = document_retrieval.search(squad[i]["question"], 3)
    # new_contexts.append(" ".join(new_context))
    new_contexts.append("asodgnaoisngaeonsdnasd")

    if i % 1000 == 0:
        print(i / len(squad))

squad = squad.add_column("new_context", new_contexts)

# save the dataset

print(squad.column_names)

squad.save_to_disk("generator/data/squad_bad_test")
