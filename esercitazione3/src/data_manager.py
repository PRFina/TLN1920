from os import name
from pathlib import Path
from collections import namedtuple
"""Note: esercitazione3/data/dd-small-nasari-15.txt is a different version from the original uploaded on moodle!
The original version contains some errors (some babel name contains ';' char, when ';' is the delimiter char!!) that have been manually fixed"""
import nltk

Document = namedtuple('Document', ['title', 'body', 'source'])

def parse_document_paragraph(document_path):
    """ Parse textual document in a semi structured named tuple. The function assume that input document have the following format:
      * First line is always the orginal source of the document marked with '#' following by one space and the actual reference.
      * Second line is always an empty line.
      * Third line is the title.
      * The remaining lines are the document actual body content.

    The body is parsed and structured at paragraph level, ie each element of body list is a paragraph of the input text.

    Args:
        document_path (pathlib.Path): path to the document file
    """
    
    with document_path.open('r') as file:
        source =  next(file)[2:].strip() # jump '# ' chars 
        next(file) # empty line after title
        title = next(file).strip()
        body = [line.strip() for line in file if line.strip() != ''] # paragraph level segmentation

    return Document(title=title, body=body, source=source)

def parse_document_sentence(document_path):
    """ Parse textual document in a semi structured named tuple. The function assume that input document have the following format:
      * First line is always the orginal source of the document marked with '#' following by one space and the actual reference.
      * Second line is always an empty line.
      * Third line is the title.
      * The remaining lines are the document actual body content.
    
    The body is parsed and structured at sentence level, ie each element of body list is a sentence of the input text.

    Args:
        document_path (pathlib.Path): path to the document file
    """
    
    with document_path.open('r') as file:
        source =  next(file)[2:].strip() # jump '# ' chars 
        next(file) # empty line after title
        title = next(file).strip()
        body = [line.strip() for line in file if line.strip() != ''] # paragraph level segmentation
        body = [sent for para in body for sent in nltk.sent_tokenize(para)] # # sentence level segmentation

    return Document(title=title, body=body, source=source)


class Nasari():
    """Class to load represent Nasari lexical resource.

    This class represent nasari as dictionary of sparse vectors:
    {key1: [(lemma1, score1), ..., (lemmak, scorek)], ...}

    Where key is the name of the word/lemma found after babel sysnset id in the nasari file;
    each item of the dictionary is a nasari vector, represented as list of tuples.
    """

    def __init__(self, nasari_path):
        """Parse a file containing nasari lexical resources and build a dictionary based representation.

        Args:
            nasari_path (pathlib.Path): file of nasari lexical resource.
        """
        self._nasari_dict = {}

        with nasari_path.open('r') as file:
            for i,line in enumerate(file.readlines()):
                parsed_chunks = line.split(';')
                babel_synset = parsed_chunks[1].lower()
                nasari_vector = []
                for chunk in parsed_chunks[2:]:
                    if len(chunk) > 1:
                        lemma, score = chunk.split('_')
                        nasari_vector.append((lemma.lower(), float(score))) # normalize text with lower()
                        
                self._nasari_dict[babel_synset] = nasari_vector # build dictionary for each line of the file

    def get_nasari_vector(self, synset_name):
        return self._nasari_dict[synset_name]
    
    def get_nasari_vectors(self):
        return self._nasari_dict;

    def build_context(self, tokens):
        """A nasari context is just a collection of lexical nasari vectors 
           associated with a given collection of tokens.

        Args:
            tokens (iterable): a collection of tokens.

        Returns:
            list: a list of nasari vectors, one for each input token. 
        """
        return [self._nasari_dict[token] for token in tokens if token in self._nasari_dict]
    