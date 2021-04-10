import csv
from pathlib import Path
import numpy as np
import pandas as pd
import data_manager as dm
import itertools
from utility import *
from dotenv import load_dotenv
import os
import requests

"""Script to help retrieve all babels synsets associated to lemmas and then geneerate a file to select the right sense and 
    annotate manually. The script use balebelnet API and so a key is required in .env file and takes some time to execute 
"""

if __name__ == '__main__':

    load_dotenv() # for babelnet API key

    semeval = dm.SemEval(Path('data/SemEval17_IT_senses2synsets.txt'))
    nasari = dm.Nasari(Path('data/mini_NASARI.tsv'), semeval)
    babelnet = dm.BabelNet(os.environ['BABELNET_KEY'])

    annotations = pd.read_csv(Path('data/words_annotations.tsv'), sep='\t')
    

    words1 = pd.DataFrame(annotations['word1'])
    words1['babelID'] = words1.apply(lambda x: list(semeval.get_synsets(x['word1'])), axis=1)
    words1 = words1.explode('babelID', ignore_index=True)

    words1['lemmas'] = words1.apply(lambda x: babelnet.get_synset_lemmas(x['babelID']), axis=1)

    words1.to_csv('output/word1')

    words2 = pd.DataFrame(annotations['word2'])
    words2['babelID'] = words2.apply(lambda x: list(semeval.get_synsets(x['word2'])), axis=1)
    words2 = words2.explode('babelID', ignore_index=True)

    words2['lemmas'] = words2.apply(lambda x: babelnet.get_synset_lemmas(x['babelID']), axis=1)

    words1.to_pickle('output/words1_df.pkl')
    words2.to_pickle('output/words2_df.pkl')

    words1 = pd.read_pickle(Path('output/words1_df.pkl'))
    words2 = pd.read_pickle(Path('output/words2_df.pkl'))

    senses1 = pd.read_csv('output/words1', sep=',', index_col=0, header=0, names=['lemma1','senseID1','terms1']).reset_index(drop=True)
    senses2 = pd.read_csv('output/words2', sep=',', index_col=0, header=0, names=['lemma2','senseID2','terms2']).reset_index(drop=True)
    senses = pd.concat([senses1,senses2], axis=1)
    
    senses.to_csv('output/senses_annotations.tsv', sep='\t')
