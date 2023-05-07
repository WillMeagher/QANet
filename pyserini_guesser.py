from typing import Any
from pyserini.search.lucene import LuceneSearcher
import subprocess
import os
import json

class pyserini_guesser:
    def __init__(self, json_file, size):
        self.json_file = json_file
        self.size = size
        data = self.load_data()
        self.searcher = LuceneSearcher('indexes/sample_collection_json')

    def load_data(self):
        with open(self.json_file) as read:
            mylist = []
            data = json.load(read)
            data = data['questions']
            for question in data:
                answer = question['answer']
                cat = question['category']
                dataset = question['dataset']
                difficulty = question['difficulty']
                subcategory = question['subcategory']
                query = question['text']
                dataset = "N/A" if dataset is None else dataset
                difficulty = "N/A" if difficulty is None else difficulty
                subcategory = "N/A" if subcategory is None else subcategory
                seperator = " "
                contents = seperator.join([ cat, dataset, difficulty, subcategory, query])
                mylist.append({"id": answer, 'contents': contents})
        new_data = json.dumps(mylist)
        with open('pydata/BM25_1.json', 'w') as write:
            write.write(new_data)
        cmd_str = f"python3 -m pyserini.index.lucene --collection JsonCollection --input pydata --index indexes/sample_collection_json --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw"
        subprocess.run(cmd_str, shell=True)



    def save(self): 
        pass
    def __call__(self, question: str, num_guesses: int) -> Any:
        hits = self.searcher.search(question)
        dict= json.loads(hits[0].raw)
        return dict['id'], self.searcher.search(question , num_guesses)[0].score

        
    def batch_guess(self, questions: list[str], num_guesses: int) -> list[Any]:
        pass