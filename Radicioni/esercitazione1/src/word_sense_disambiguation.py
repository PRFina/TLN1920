from nltk.corpus import wordnet as wn
import nltk
from pathlib import Path

# a set of common english stop words
STOP_WORDS = set(Path('data/stop_words_FULL.txt').open('r').read().splitlines())


def bow_model(sentence, stopwords=None):
    """Build a bag of word (BOW) model for a given sentence, 
    removing puntcuation marks and optionally stopwords. 

    Args:
        sentence (string): sentence from which build BOW.
        stopwords (set, optional): set of stopwords to remove from BOW. Defaults to None.

    Returns:
        set: bag of word.
    """
    bow = set(nltk.word_tokenize(sentence))
    punct = {',', ';', '(', ')', '{', '}', ':', '?', '!'} 
    
    bow = bow.difference(punct)
    if stopwords:
        bow = bow.difference(stopwords)

    return bow

def lesk_wsd(sentence, ambiguous_word, stopwords=None):
    """ Lesk word sense disambiguation algorithm. Given ambiguous word, the algorithm use the
    sentence as disambiguation context and use wordnet lexical information to find the
    best sense signature that maximally overlap with context. 

    Both context and sense signature use a bag-of-word model representation. Sense signature
    use both synset gloss definition and eventually example sentences.

    Args:
        sentence (string): a single sentence containing the ambiguous word.
        ambiguous_word ([type]): ambiguous/polysemous word to disambiguate.
        stopwords ([type], optional): a set of stop words to remove. Defaults to None.

    Returns:
        (wordnet synset, integer): the best sense wordnet synset and its overlap metric value
    """
    
    best_sense = None
    max_overlap = 0

    context = bow_model(sentence)

    for syn in wn.synsets(ambiguous_word): # foreach sense
        signature  = bow_model(syn.definition(), stopwords) # gloss words
        
        for example in syn.examples(): # examples words
            signature.union(bow_model(example, stopwords)) 
        
        overlap = len(context.intersection(signature))
        if overlap > max_overlap: # > returns the first best sense in case of overlap ties
            max_overlap = overlap
            best_sense = syn

    return best_sense, max_overlap