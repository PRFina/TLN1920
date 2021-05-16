
from pathlib import Path
import numpy as np

from nltk.tokenize import RegexpTokenizer
import re

class Document:
    BLOCK_SEPARATOR = "========"
    #WIKI_727K_SEPARATOR_REGX = "========,*[0-9]*,*[\w+\s*]*\.+"
    # match anything in between 8 consecutives '=' and a newline 
    WIKI_727K_SEPARATOR_REGX = "========.\n*" 

    def __init__(self, doc_file, tokenizer, search_title=False):
        self._chunk_tokenizer = tokenizer
        self._section_tokenizer = RegexpTokenizer(Document.WIKI_727K_SEPARATOR_REGX, gaps=True,
                                                  flags=re.UNICODE | re.MULTILINE)

        
        with doc_file.open() as file:
            self._sections = self._section_tokenizer.tokenize(file.read())
            sections = [section.strip() for section in self._sections] # remove newlines
            
            if search_title:
                self.title = sections[0]
                self.body_sections = sections[1:]
            else:
                self.title = "unknow document title"
                self.body_sections = sections
            
            self._chunks = []

            for section in self.body_sections:
                self._chunks.extend(self._chunk_tokenizer(section))
                self._chunks.append(Document.BLOCK_SEPARATOR)
            self._chunks = self._chunks[0:-1] # remove last separator
    
    def get_chunks(self):
        return [chunk for chunk in self._chunks 
                if chunk is not self.BLOCK_SEPARATOR]
    
    def get_breakpoints(self):
        sep_pos = [i for i,chunk in enumerate(self._chunks) 
                   if chunk == Document.BLOCK_SEPARATOR]
        
        breakpoints_pos = [(pos-1, pos) for pos in sep_pos] # we must take into account that pos when BLOCK_SEPARATOR is removed, is the right index of the next sentence (not pos+1)

        return breakpoints_pos



class Glove:
    """Simple wrapper to parse and load Glove Embedding reosurce"""
    
    def __init__(self, pretrained_file):
        self.embedding = {}

        with pretrained_file.open() as glove_file:
            for line in glove_file:
                cols = line.split(" ")
                self.embedding[cols[0]] = np.array(cols[1:], dtype=np.float)
    
    def __getitem__(self,key):
        return self.embedding.get(key, None)