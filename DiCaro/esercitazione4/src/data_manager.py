
from pathlib import Path
import numpy as np

class Document:
    BLOCK_SEPARATOR = "$$^^$$"

    def __init__(self, doc_file, tokenizer):
        self.tokenizer = tokenizer

        with doc_file.open() as file:
            paragraphs = file.read().split('\n\n') # we implicitely assume that double paragraph is the "golden label" for topic separation
            self.title = paragraphs[0]
            self.body_paragraphs = paragraphs[1:]
            self._chunks = []

            for para in self.body_paragraphs:
                self._chunks.extend(self.tokenizer(para))
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