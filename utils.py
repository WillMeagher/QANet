import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

## You may need to download stopwords and wordnet with the following: nltk.download('wordnet')

class utils:
## Text Preprocessing
    def clean_text(text: str) -> str:
        """
        This will lowercase the text, remove punctuation, remove stopwords, and lemmatize.
        return: str
        """
        # Lowercase
        text = text.lower()
        # Remove stopwords
        text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
        # Remove punctuation
        text = ''.join([char for char in text if char not in punctuation])
        # Lemmatize
        lemma = WordNetLemmatizer()
        text = ' '.join([lemma.lemmatize(word) for word in text.split()])
        return text


    def tokenize_text_words(text: str) -> list[str]:
        """
        This will tokenize the text into words.
        return: List[str]
        """
        return word_tokenize(text)

    def tokenize_text_sentences(text: str) -> list[str]:
        """
        This will tokenize the text into sentences.
        return: List[str]
        """
        return sent_tokenize(text)

## General Utils
    def get_top_n(lst: list, n: int) -> list:
        """
        This will sort the list then return the top n items from a list.
        return: List
        """
        lst.sort(reverse = True)
        return lst[:n]
        
