import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
import numpy as np
from pathlib import Path
import itertools as it

from  sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(definition):
    """Apply pre-processing logic to raw string sentence

    Args:
        definition (str): raw string representation of the concept definition

    Returns:
        Set of Str: bag-of-word representation of the input definition
    """

    # Tokenization
    definition = definition.lower()
    tokens = set(nltk.word_tokenize(definition))

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    punct = {',', ';', '(', ')', '{', '}', ':', '?', '!', '.', "'s"}    
    tokens = tokens.difference((stop_words.union(punct)))

    # Lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = set(lemmatizer.lemmatize(token) for token in tokens)

    return lemmatized_tokens


def overlap_similarity(bow1, bow2):
    """ Compute similarity between two concept definition using bag-of-words representation.
       The similarity is assessed by interection over the minimum length between the two BOW

    Args:
        bow1 (Set of Str): BOW representation of the first concept.
        bow2 (Set of Str): BOW representation of the second concept.

    Returns:
        float: similarity score
    """
    return len(bow1 & bow2) / min(len(bow1), len(bow2))


def concept_matrix_similarity(definitions, preprocess_func, sim_func):
    """ Compute for each column (a concept) of the dataframe
        the pairwise similarity among all possible definition combinations.

        For each concept, all pairwise similarity scores are mean aggregated.

       
    Args:
        definitions (pandas.DataFrame): dataframe containing the concepts definitions
        preprocess_func (callable): callable to preprocess raw string definitions
        sim_func (callable): callable to compute the similarity between two concepts

    Returns:
        tuple(Dict, numpy.ndarray): first tuple element: dictionary with keys as dataframe 
        column names and values as avg similarity scores. Second tuple element an numpy.ndarray
        of shape (2,2) with values from Dict.values()
    """
    defs_df = definitions.copy() # don't override the original dataframe
    avg_scores = {}

    for col in defs_df.columns:
        # apply preprocessing function
        defs_df[col] = defs_df.apply(lambda x: preprocess_func(x[col]), axis=1)
        # compute similarites for every definitions combinations pairs
        sim_scores = np.array([sim_func(def1,def2) for def1, def2 in it.combinations(defs_df[col], 2)])
        # compute avg similarity
        avg_scores[col] = sim_scores.mean()
    
    return avg_scores, np.array(list(avg_scores.values())).reshape((2,2))


## VSM representation functions

def vsm_preprocess(definition, vectorizer):
    """ Apply vectorizer trasformation to the input raw string representation

    Args:
        definition (str): raw string of the concept definition 
        vectorizer (sklearn._mixinVectorizer): a vectorizer instance

    Returns:
        [type]: [description]
    """
    return vectorizer.transform([definition])


def concept_matrix_similarity_vsm(definitions, preprocess_func, sim_func):
    """ Compute for each column (a concept) of the dataframe
        the pairwise similarity among all possible definition combinations.

        For each concept, all pairwise similarity scores are mean aggregated.

        This function is specific for VSM since internally instantiate a Vectorizer instance
        and fit it on concept definitions that act like a "corpus".
       
    Args:
        definitions (pandas.DataFrame): dataframe containing the concepts definitions
        preprocess_func (callable): callable to preprocess raw string definitions
        sim_func (callable): callable to compute the similarity between two concepts

    Returns:
        tuple(Dict, numpy.ndarray): first tuple element: dictionary with keys as dataframe 
        column names and values as avg similarity scores. Second tuple element an numpy.ndarray
        of shape (2,2) with values from Dict.values()
    """
    defs_df = definitions.copy()
    avg_scores = {}

    for col in defs_df.columns:
        # per-concept vectorization 
        vectorizer = CountVectorizer(analyzer='word', stop_words='english').fit(defs_df[col])
        # apply preprocessing function
        defs_df[col] = defs_df.apply(lambda x: preprocess_func(x[col], vectorizer), axis=1)
        # compute similarites for every definitions combinations pairs
        sim_scores = np.array([sim_func(def1,def2) for def1, def2 in it.combinations(defs_df[col], 2)])
        # compute avg similarity
        avg_scores[col] = sim_scores.mean()
    
    return avg_scores, np.array(list(avg_scores.values())).reshape((2,2))
