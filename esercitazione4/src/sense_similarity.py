import numpy as np
from scipy.spatial import distance
import itertools


def sense_similarity(word1, word2, similarity_func, nasari):
    """ Compute the sense similarity as maximum among 
        all possible pair of senses of the given words.

    Args:
        word1 (str): first word
        word2 (str): second word
        similarity_func (callable): similarity function with (numpy.ndarray, numpy.ndarray) -> float signature.
        nasari (data_manger.Nasari): a Nasari lexical resource instance 

    Returns:
        (float, str,str): return a triple (score, babel synset id, babel synset id) with maximal similarity and the sense pair that maximize the similarity.
    """
    w1_vectors = zip(nasari.get_lemma_senses(word1), 
                     nasari.get_lemma_vectors(word1)) # all senses pair of (synset id, nasari vector)
    w2_vectors = zip(nasari.get_lemma_senses(word2), 
                     nasari.get_lemma_vectors(word2)) # all senses pair of (synset id, nasari vector)
    
    scores = []
    senses = []

    for (id1, v1), (id2, v2) in itertools.product(w1_vectors, w2_vectors): # generate cartesian product of all possible pairs of senses
        if not(v1 is None or v2 is None): # skip if one or both of v1,v2 are none
            scores.append(similarity_func(v1, v2))
            senses.append((id1,id2))
    
    max_score = None
    max_senses = (None,None)

    if len(scores):
        max_idx = np.argmax(scores)
        max_score = scores[max_idx]
        max_senses = senses[max_idx]
    
    return max_score, max_senses[0], max_senses[1] # (score, word1, word2) instead of (score, (word1,word2))


def sense_similarity_score(word1, word2, similarity_func, nasari):
    """Just an helper function wrapping sense_similarity function. Compute only the similarity score without senses.
    Args:
        word1 (str): first word
        word2 (str): second word
        similarity_func (callable): similarity function with (numpy.ndarray, numpy.ndarray) -> float signature.
        nasari (data_manger.Nasari): a Nasari lexical resource instance 

    Returns:
        float: maximal similarity score among all possible pairs of senses
    """

    return sense_similarity(word1, word2, similarity_func, nasari)[0] # return only score and discard senses

def cosine_similarity(v1, v2):
    """ Compute the cosine similarity metric between two vectors

    Args:
        v1 (numpy.ndarray): vector 1
        v2 (numpy.ndarray): vector 2

    Returns:
        [float]: similarity score in [0,1] range.
    """
    return 1-distance.cosine(v1,v2) # since scipy function is a distance