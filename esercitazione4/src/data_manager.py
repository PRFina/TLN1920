from os import name
from pathlib import Path
from collections import namedtuple
import nltk
import csv
import numpy as np
import requests

class LemmaToSensesMapper():
    """ interface that represent a 1-to-n mapping between a given lemma and associated senses.
    Essentally a mapper is wrapper around a dictionary to represent the mapping lemma-senses due to polysemy.

    The dictionary should have the following structure:
    
    {key:list of senses}
    
    where:
        key should be a lemma
        list of senses: a list of some representation for the senses (vectors, synset Id, etc.)
    """
    
    def __init__(self) -> None:
        self._lemmas = {} # should be dict {key: list of senses}
    
    def get_synsetsID(self, lemma):
        pass

class SemEval(LemmaToSensesMapper):
    """Class to load and access SemEvalIta2017 annotations. 
    Annotation file contains polysemous lemmas and an abritary number of associated babel senses,

    with the following structure:

    #lemma\n
    <babel synset id#1>\n
    <babel synset id#2>\n
    ....     
    <babel synset id#k>\n
    
    eg:

    #Rinascimento
    bn:00066458n
    bn:15363387n
    bn:00067098n
    bn:00044080n
    """
        
    def __init__(self, semeval_path) -> None:
        super().__init__()

        with semeval_path.open('r') as file:
            blocks = file.read().split('#') # (word, [synsets]) are delimited by # char
            for lemma_block in blocks[1:]: # first element is always an empty string
                lines = lemma_block.splitlines()
                synset_word = lines[0] # first line is the lemma, the remaining ones are babel IDs
        
                self._lemmas[synset_word] = set(lines[1:]) # babel synsets IDs

    def get_synsetsID(self, lemma):
        """ Get a list of associated babel senses with a given lemma 

        Args:
            lemma (str): lemma (word)

        Returns:
            [list of str]: list of babel synset IDs
        """
        return self._lemmas[lemma]

class Nasari():
    """ Class to load and access Nasari embedded version.

        Each lemma in nasari have one or more vector. Each vector is obtained from an embedding procedure 
        and so is a numerical vector.


    """

    def __init__(self, nasari_path, mapper=None):
        """Build a nasari instance with an optional LemmaToSensesMapper instance.

        Args:
            nasari_path (pathlib.Path): [description]
            mapper (LemmaToSensesMapper, optional): a LemmaToSensesMapper istance . Defaults to None.
        """
        self._nasari = {}
        self._nasari_words = {}
        self._mapper = mapper

        with nasari_path.open('r') as file:
            tsv_reader = csv.reader(file, delimiter='\t') # for tsv files
            
            for row in tsv_reader:
                babelID, synset_word = row[0].split('__') # first element is always synsetID__word
                vector = np.array(row[1:]).astype('float')
                        
                self._nasari[babelID] = vector # build dictionary for each row of the file
                self._nasari_words[babelID] = synset_word


    def get_vector(self, synsetID):
        """ Given an input sense, represented with a babelnet synset id, get the associated nasari embedded vector. 

        Args:
            synsetID (str): babel synset id

        Returns:
            numpy.ndarray : Nasari embedded vector of the given sense
        """
        return self._nasari.get(synsetID)

    def get_lemma_vectors(self, lemma):
        """ Given an input lemma get all the associated nasari embedded vectors. 
            This method requires a LemmaToSensesMapper instance!

        Args:
            lemma (str): the given lemma/word to search for

        Raises:
            TypeError: if no LemmaToSensesMapper instance is given in the object constructor.

        Returns:
            [list of numpy.ndarray]: list of Nasari embedded vectors
        """
        if not self._mapper:
            raise  TypeError('To use this method you must assign a mapper in the object initilization')
        return [self.get_vector(synID) for synID in self._mapper.get_synsetsID(lemma)]
    
    def get_lemma_senses(self, lemma):
        """ Given an input lemma get all the associated senses as babel synset IDs. 
            This method requires a LemmaToSensesMapper instance!

        Args:
            lemma (str): the given lemma/word to search for

        Raises:
            TypeError: if no LemmaToSensesMapper instance is given in the object constructor.

        Returns:
            [list of str]: list of babel synset IDs
        """
        if not self._mapper:
            raise  TypeError('To use this method you must assign a mapper in the object initilization')
        return [synID for synID in self._mapper.get_synsetsID(lemma)]


class BabelNet():
    """Minimal wrapper to BabelNet API.
    """

    def __init__(self, api_key):
        """[summary]

        Args:
            api_key (str): key to get access to the API. Request it at https://babelnet.org/register
        """
        self._API_KEY = api_key
        self._endpoint_URL = "https://babelnet.io/v5/getSynset" # API endpoint to retrieve synsets


    def get_synset_lemmas(self, babelId, lang='IT'):
        """Retrieve all associated terms for the given babel sense.
        Warning: this method make a call to the BabelNet API and execution can be slow if called on batch data!
        Furthermore, each method call consume one babel coin.

        Args:
            babelId (str): babel synset ID
            lang (str, optional): target language for the search. Defaults to 'IT'.

        Returns:
            [set]: set of terms of for the given sense.
        """
        params = {
                'id': babelId,
                'key': self._API_KEY,
                'targetLang': lang
        }

        response = (requests.get(url=self._endpoint_URL , params=params)
                            .json())
        senses = response.get('senses')
        lemmas = None

        if senses and len(senses):
                lemmas = set([sense['properties']['fullLemma'] for sense in senses]) # no duplicates

        return lemmas    