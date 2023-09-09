from nltk.corpus.reader import nltk
from typing import List
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import string
import re

NUMBER_PRESENT = re.compile(r'\d+')

def get_wordnet_tag(tag):
    if tag.startswith('J'):
      return wordnet.ADJ
    elif tag.startswith('V'):
      return wordnet.VERB
    elif tag.startswith('N'): 
      return wordnet.NOUN
    elif tag.startswith('R'):
      return wordnet.ADV
    return wordnet.NOUN


def get_tokens_and_tags(raw_text: str) -> List:
    text = raw_text.lower().translate(str.maketrans('', '', string.punctuation))

    words_and_tags = nltk.pos_tag(text.split())

    return list(map(lambda x: (x[0], get_wordnet_tag(x[1])), words_and_tags))

def clean_text(text: str, strip_numbers: bool = False) -> List:
    clean_tokens: List[str] = []
    tokens_and_tags: List[str] = get_tokens_and_tags(text)

    lemmatizer = WordNetLemmatizer()
    en_stopwords = set(stopwords.words('english'))

    for token, tag in tokens_and_tags:
        token = token.strip() # remove space
        if strip_numbers and NUMBER_PRESENT.search(token):
            continue
        if token not in en_stopwords and token != '•':
            lemmatized_token = lemmatizer.lemmatize(token, pos=tag)
            clean_tokens.append(lemmatized_token)
    
    return ' '.join(clean_tokens)