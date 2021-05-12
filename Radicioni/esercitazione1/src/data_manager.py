import csv
from pathlib import Path
import re
import pandas as pd
import xml.etree.ElementTree as ET
from lxml import etree as Exml
from nltk.corpus import wordnet as wn


class WordSimCorpus:
    """
    Simple wrapper for wordSim353 corpus. This class implement the iterator interface to loop over
    triples of (word1, word2, similarity score).

    The similarity score is rescaled from [0,10] range to [0,1].
    """
    
    def __init__(self, corpus_path):
        self._corpus_file = open(corpus_path,'r')
        self._csv_reader = csv.reader(self._corpus_file, delimiter=',')

    def __iter__(self):
        self._csv_reader.__next__() # skip header line
        return self
    
    def __next__(self):
        x = self._csv_reader.__next__() 
        return (x[0], x[1], float(x[2])/10.0)  # rescale gold standard value from [0,10] to [0,1]

    def get_word_pairs(self):
        next(self._csv_reader) # skip header line
        return [(line[0],line[1]) for line in self._csv_reader]
    
    def get_gold_standard_scores(self):
        next(self._csv_reader) # skip header line
        return [float(line[2])/10 for line in self._csv_reader] # rescale gold standard value from [0,10] to [0,1]
    
    def __del__(self):
        self._corpus_file.close()
            


class WSDSentences:
    """
    Simple wrapper for reading txt files where each line is a sentence 
    and polysemous words are marked in: **<word>** (double asterisks).
   
    One foundamental assumption behind the class methods working logic:
    there is only one polysemous word for sentence! 

    The class mantains a dictionary instance to keep track of sentences,
    and polysemous word of each sentence.
    """
    
    def __init__(self, corpus_path):
        self._sentences = {}
        self._target_RE = '\*\*(.*)\*\*'

        with corpus_path.open('r') as file:
            for idx, line in enumerate(file.readlines()):
                # match and extract polysemous word
                poly_word = re.search(self._target_RE, line, re.IGNORECASE).group(1) 
                self._sentences[idx] = (poly_word,line.rstrip())
    
    def get_sentences(self):
        """
        Helper method to remove marked symbols from sentences.
        Returns:
            List of triples: return list of triples of (key, polysemous word, sentence)
        """

        return [(key, poly_word, sentence.replace('**','')) for 
                 key, (poly_word, sentence) in self._sentences.items()]

    def replace_polysemous_word(self, key, repl_string):
        """Replace polysemous word with repl_string

        Args:
            key (int): key of sentence to replace
            repl_string (str): new string to which to replace 

        Returns:
            (str): sentence with replaced text 
        """
        sent_to_repl = self._sentences[key][1]
        new_sent = re.sub(self._target_RE, repl_string, sent_to_repl)

        return new_sent


class SemCorCorpus():

    def __init__(self, brown_file_path):
        data = None
        with open(brown_file_path, 'r') as fileXML:
            data = fileXML.read()

            # correct bad formatting xml 
            data = data.replace('\n', '')
            replacer = re.compile("=([\w|:|\-|$|(|)|']*)")
            data = replacer.sub(r'="\1"', data)

        self._xml_data = data

    def get_annotated_sentences(self, wordnet_pos='n'):
        result = []
        try:
            paragraphs = Exml.XML(self._xml_data).findall("./context/p") # find paragraphs
            sentences = []
            for p in paragraphs:
                sentences.extend(p.findall("./s")) # find sentences
            
            for sentence in sentences:
                word_tags = sentence.findall('wf')
                sentence_words = []
                tuple_list = []
                
                for word_tag in word_tags:
                    word_form = word_tag.text # word form
                    pos = word_tag.attrib['pos'] # word form part of speech
                    sentence_words.append(word_form) # collect sentence words 

                    # select only nouns (NN), no named entities (_), polysemous and with an associated synset
                    if pos == 'NN' and '_' not in word_form and len(wn.synsets(word_form)) > 1 and 'wnsn' in word_tag.attrib:
                        synset_id = self._build_synset_id(word_form, word_tag.attrib['wnsn'], wordnet_pos)
                        tuple_list.append((word_form, synset_id))
                
                sentence = ' '.join(sentence_words) # concatenate words to build the sentence
                result.append((sentence, tuple_list))

        except Exception as e:
            raise NameError('xml: ' + str(e))
    
        return result

    def _build_synset_id(self, word, sense_n, wordnet_pos='n'):
        return "{}.{}.{:02}".format(word, wordnet_pos, int(sense_n))



if __name__ == '__main__':

    pass