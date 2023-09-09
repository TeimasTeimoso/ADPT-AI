from gensim.models import KeyedVectors
import gensim.downloader
import numpy as np
import pandas as pd
from typing import List

#google_news_word2vec = KeyedVectors.load_word2vec_format('/home/rjft/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz', binary=True)
google_news_word2vec = gensim.downloader.load('word2vec-google-news-300')

class Word2VecMean:
    def __init__(self, model=google_news_word2vec) -> None:
        self.model = model
        self.vec_size = self.model.vector_size

    # def fit_transform(self, text_df: pd.DataFrame) -> np.ndarray:
    #     return text_df.apply(self.__sent_vec).to_numpy()

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame =None) -> 'Word2VecMean':
        return self

    def transform(self, X: pd.DataFrame) -> List[np.float32]:
        return X.apply(self.__sent_vec).to_list()

    def set_params(self, **params):
        return super().set_params(**params)

    def __sent_vec(self, sent: str) -> np.ndarray:
        ctr = 1
        sent = self.__get_tokens(sent)
        wv_res = np.zeros(self.vec_size)

        for word in sent:
            if word in self.model:
                wv_res += self.model[word]
                ctr += 1

        wv_res = wv_res / ctr

        return wv_res

    @staticmethod
    def __get_tokens(str: str) -> List[str]:
        return str.split()
