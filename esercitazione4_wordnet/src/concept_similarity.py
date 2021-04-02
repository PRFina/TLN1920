import numpy as np
import nltk as nl
import math
from nltk.corpus.reader.wordnet import WordNetError
from nltk.corpus import wordnet as wn

"""
Warning: since WordSim353 contains only noun pairs there's no need to
simulate_root since from wordnet >1.6 noun taxonomy have only one root (entity.n.01)
"""
def wu_palmer_similarity(synset1, synset2):
    """ Concept similarity algorithm based on Wu-Palmer formula.

    Args:
        synset1 (wordnet sysnset): first sense
        synset2 (wordnet sysnset): second sense

    Returns:
        [float]: normalized [0,1] similarity score. 0 for no similarity at all, 1 for same senses.
    """
    subsumers = synset1.lowest_common_hypernyms(synset1, use_min_depth=True, simulate_root=False)
    
    if len(subsumers) == 0: # no LCS found
        return None 
    
    lcs = subsumers[0] # take just the first one among all possible LSC
    depth_lcs = lcs.max_depth() + 1
    
    len1 = synset1.shortest_path_distance(lcs, simulate_root=False)
    len2 = synset2.shortest_path_distance(lcs, simulate_root=False)
    
    if len1 is None or len2 is None:
            return None

    return (2.0 * depth_lcs) / (len1 + len2 + 2*depth_lcs)


def shortest_path_similarity(synset1, synset2):
    """ Concept similarity algorithm based on the difference between the 
    shortest path of the two senses and the max depth of wordnet taxonomy.
    
    The first execution could take some time since POS-specific taxonomy max depth
    must be computed (if not already computed). 

    Args:
        synset1 (wordnet sysnset): first sense
        synset2 (wordnet sysnset): second sense

    Returns:
        [float]: normalized [0,1] similarity score. 0 for no similarity at all, 1 for same senses.
    """
    if synset1.pos() != synset2.pos():
            raise WordNetError(
                "Computing the similarity requires {} and {} to have the same part of speech.".format(synset1, synset2))   

    max_depth = get_taxonomy_max_depth(synset1)
    dist = synset1.shortest_path_distance(synset2)
    
    if dist is None:
        similarity = 0
    else:
        similarity = 2 * max_depth - dist
    
    return similarity / (2 * max_depth) # normalize similarity in [0,1] range


def leakcock_chodorow_similarity(synset1, synset2):
    """ Concept similarity algorithm based on the Leakcock-Chodorow formula.
    
    The first execution could take some time since POS-specific taxonomy max depth
    must be computed (if not already computed). 

    Args:
        synset1 (wordnet sysnset): first sense
        synset2 (wordnet sysnset): second sense

    Returns:
        [float]: similarity score in range [0, inf]
    """
    
    if synset1.pos() != synset2.pos():
        raise WordNetError(
            "Computing the similarity requires {} and {} to have the same part of speech.".format(synset1, synset2))

    max_depth = get_taxonomy_max_depth(synset1)
    dist = synset1.shortest_path_distance(synset2)
    if dist is None:
        imilarity = 0
    else:
        similarity = -math.log((dist+1)/(2*max_depth + 1))

    return similarity 


def get_taxonomy_max_depth(synset):
    """Helper function to retrieve max depth of the corresponding sysnet POS taxonomy

    Args:
        synset (wordnet synset): synset used to retrieve the associated POS.

    Returns:
        int: maximum depth of the taxonomy
    """
    taxonomies_depth = synset._wordnet_corpus_reader._max_depth
    if synset.pos() not in taxonomies_depth:
        synset._wordnet_corpus_reader._compute_max_depth(synset.pos(), simulate_root=True)

    taxonomy_max_depth = taxonomies_depth[synset.pos()]

    return taxonomy_max_depth

def word_similarity(word1,word2, similarity_func, pos='n'):
    """Compute the concept similarity as maximum among 
    all possible pair of senses of the given words.

    Args:
        word1 (string): first word
        word2 ([type]): second word
        similarity_func (function): similarity function with (synset,synset) -> float signature.
        pos (str, optional): wordnet supported part-of-speech to restrict the taxonomy to be searched. Defaults to 'n'.

    Returns:
        float: similrity score
    """
    similarities = [similarity_func(s1,s2) for s1 in wn.synsets(word1, pos) for s2 in wn.synsets(word2, pos)]
    if all(sim == None for sim in similarities):
        cs = None
    else:
        cs = max(sim for sim in similarities if sim is not None)
    return cs