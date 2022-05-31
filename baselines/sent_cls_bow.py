"""
Bag-of-words model for sentence classification.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

class BagOfWords:
    """
    BOW + Naive Bayes
    """
    def __init__(self,
        classifier,
        ngram_range,
        max_features,
        min_df,
    ):
        self.encoder = TfidfVectorizer(
            ngram_range = ngram_range,
            min_df = min_df,
            max_features = max_features
        )

    def run(self):
        pass

