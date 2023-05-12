# THIS DOES NOT WORK
# Something with the object being immutable probably
import document_retrieval
from datasets import load_dataset

squad = load_dataset("squad", split="train[:1000]")
squad = squad.train_test_split(test_size=0.2)

# created deep copy of squad
squad = squad.copy()

# try using .add_coloumn()

sets = ["train", "test"]
for set in sets:
    for i in range(len(squad[set])):
        new_context = document_retrieval.search(squad[set][i]["question"], 3)
        new_question = squad[set][i]
        new_question["context"] = " ".join(new_context)
        

        print(squad[set][i]["context"])

        if i % 1000 == 0:
            print(i / len(squad[set]))


# save the dataset

print(squad["train"][0])

squad.save_to_disk("squad_new")
exit()
