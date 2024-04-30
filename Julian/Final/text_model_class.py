import numpy as np
import opencc
import gensim
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords, names
import enchant
from sklearn.neighbors import NearestNeighbors
import nltk
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = stopwords.words("english")
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

def load_list_from_file(file_name):
    try:
        with open(file_name, 'rb') as file:
            lst = pickle.load(file)
        return lst
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    
def preprocess(paragraphs):
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()

    processed_docs = []

    for paragraph in paragraphs:
        words = gensim.utils.simple_preprocess(paragraph, min_len=3, deacc=True)
        lemmatized_words = [LEMMATIZER.lemmatize(word) for word in words if word not in STOP_WORDS]

        processed_doc = " ".join(lemmatized_words)
        processed_docs.append(processed_doc)

    return processed_docs

class TextModel:
    def __init__(self, file_name, vectorization='lsa', dimension=200, min_df=2):
        self.vectorization = vectorization
        loaded_paragraphs = load_list_from_file(file_name)
        if loaded_paragraphs is not None:
            self.paragraphs = loaded_paragraphs
        else:
            self.paragraphs = []
        self.preprocessed_paragraphs = preprocess(self.paragraphs)
        
        if self.vectorization == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=min_df)
            self.vector_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
        elif self.vectorization == 'lsa':
            self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_paragraphs)
            self.svd = TruncatedSVD(n_components=dimension, algorithm='randomized')
            self.vector_matrix = self.svd.fit_transform(self.tfidf_matrix)

    def test(self):
        print("Placeholder for testing")

    def find_similar_items(self, query, top_n=30):
        if self.vectorization == 'tfidf':
            query_vector = self.tfidf_vectorizer.transform([query])
        elif self.vectorization == 'lsa':
            query_vector = self.tfidf_vectorizer.transform([query])
            query_vector = self.svd.transform(query_vector)

        cosine_similarities = linear_kernel(query_vector, self.vector_matrix).flatten()
        relevant_items_indices = cosine_similarities.argsort()[-top_n:][::-1]
        relevant_items = [self.paragraphs[i] for i in relevant_items_indices if i < len(self.paragraphs)]
        return relevant_items

    def vectorize(self, query):
        if self.vectorization == 'lsa':
            processedQuery = preprocess([query])[0]
            tfidf_query = self.tfidf_vectorizer.transform([processedQuery])
            query_vector = self.svd.transform(tfidf_query)
            return query_vector
        elif self.vectorization == 'tfidf':
            processedQuery = preprocess([query])[0]
            query_vector = self.tfidf_vectorizer.transform([processedQuery])
            return query_vector

    def get_key_words(self, v, n=10):
        if self.vectorization == 'lsa':
            v = self.svd.inverse_transform(v)[0]
        top_indices = np.argpartition(v, -n)[-n:]
        top_indices_sorted = top_indices[np.argsort(v[top_indices])[::-1]]  # Sort indices by highest values
        words = self.tfidf_vectorizer.get_feature_names_out()
        return [words[i] for i in top_indices_sorted]