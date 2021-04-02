from  concept_similarity import word_similarity, wu_palmer_similarity, leakcock_chodorow_similarity,shortest_path_similarity
from pathlib import Path
import pandas as pd
from nltk.corpus.reader.wordnet import wup_similarity, path_similarity, lch_similarity
from WordSim import WordSimCorpus

""""
script to test custom implementation 
wrt nltk reference implementation of some wordnet synset similarity measures.

The ouput is a descriptive statistics summary between the absolute differences 
of scores obtained with custom and nlkt implementations over the wordsim353 
words pairs.

The question is: how much the results of the two implementations differs?

Warning: NLTK and custom path similarity implements a different formula so results must not be taken in consideration!
"""

if __name__ == '__main__':

    custom_metrics = [(wu_palmer_similarity, 'wu_palmer'), 
                      (shortest_path_similarity, 'shortest_path'),
                      (leakcock_chodorow_similarity, 'lch')]
    
    nltk_metrics = [(wup_similarity, 'wu_palmer'), 
                    (path_similarity, 'shortest_path'),
                    (lch_similarity, 'lch')]

    nltk_stats = pd.DataFrame({'gold_standard': WordSimCorpus(Path('data/WordSim353.csv')).get_gold_standard_scores()})
    custom_stats = pd.DataFrame({'gold_standard': WordSimCorpus(Path('data/WordSim353.csv')).get_gold_standard_scores()})
    
    for metric, name in custom_metrics:
        ws353 = WordSimCorpus(Path('data/WordSim353.csv'))
        scores = [word_similarity(w1,w2, metric) for w1,w2,gold_val in ws353]
        custom_stats[name] = scores

    for metric, name in nltk_metrics:
        ws353 = WordSimCorpus(Path('data/WordSim353.csv'))
        scores = [word_similarity(w1,w2, metric) for w1,w2,gold_val in ws353]
        nltk_stats[name] = scores
    
    (print("Descriptive stats of absolute differences between custom and nltk implementations\n{}"
         .format((custom_stats-nltk_stats).abs().drop('gold_standard',axis=1).describe())))