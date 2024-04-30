import nltk
import os
import pickle
import enchant
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import opencc
# Download necessary NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('words')
nltk.download('stopwords')

# Initialize global variables
ENGLISH_DICT1 = enchant.Dict("en_UK")
ENGLISH_DICT2 = enchant.Dict("en_US")
STOP_WORDS = stopwords.words("english")
LEMMATIZER = WordNetLemmatizer()

def preprocess(paragraphs):
    """ Preprocess a list of paragraphs by removing stopwords and applying lemmatization. """
    processed_docs = []
    for paragraph in paragraphs:
        words = simple_preprocess(paragraph, min_len=3, deacc=True)
        lemmatized_words = [LEMMATIZER.lemmatize(word) for word in words if word not in STOP_WORDS]
        processed_doc = " ".join(lemmatized_words)
        processed_docs.append(processed_doc)
    return processed_docs

# NLTK and other required modules are assumed to be imported and initialized as in the previous example

class SubtitleProcessor:
    def __init__(self, merged_subtitles_path):
        self.merged_subtitles_path = merged_subtitles_path
        self.merged_subtitles = self.load_subtitles(self.merged_subtitles_path)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.build_tfidf_matrix()

    def load_subtitles(self, file_path):
        """ Load subtitles from a pickle file. """
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def build_tfidf_matrix(self):
        """ Build a TF-IDF matrix for all merged subtitles. """
        processed_texts = preprocess([' '.join(sub['text']) for sub in self.merged_subtitles])
        return self.vectorizer.fit_transform(processed_texts)

    def find_similar(self, query, top_n=9):
        """ Find top_n similar subtitles to the query. """
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        similar_indices = np.argsort(-cosine_similarities)[:top_n]
        return [(index, self.merged_subtitles[index], cosine_similarities[index]) for index in similar_indices]

