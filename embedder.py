import numpy as np
import gensim
"""
A Word2Vec embedder class
"""

class Word2VecEmbedder:
    def __init__(self, sentances, vector_size=100, window=5, min_count=1, workers=4):
        self.model = gensim.models.Word2Vec(sentances, vector_size=vector_size, window=window,
                                            min_count=min_count, workers=workers)
        self.vector_size = vector_size

    def embed(self, sentance, use_similarity=False):
        embedded_sentance = []
        for word in sentance:
            if word in self.model.wv:
                embedded_sentance.append(self.model.wv[word])
            else:
                if use_similarity:
                    # Find the most similar word in the vocabulary
                    similar_words = self.model.wv.most_similar(positive=[word], topn=1)
                    if similar_words:
                        embedded_sentance.append(self.model.wv[similar_words[0][0]])
                    else:
                        embedded_sentance.append(np.zeros(self.vector_size))
                else:
                    embedded_sentance.append(np.zeros(self.vector_size))
        return embedded_sentance

    def save_model(self, file_path):
        self.model.save(file_path)
