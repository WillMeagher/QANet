from pyserini.search.lucene import LuceneSearcher
import json

searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')

def search(query, k=10):
    hits = searcher.search(query, k=k)
    hits = [json.loads(hits[i].raw)['contents'] for i in range(len(hits))]
    return hits

def main():
    query = 'What was the name of the short-lived Marvel novelization book publisher during the 2000s?'
    hits = search(query)

    for hit in hits:
        print(hit)

if __name__ == '__main__':
    main()