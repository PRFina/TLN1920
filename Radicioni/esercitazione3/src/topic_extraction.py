import nltk
from nltk.corpus import stopwords
from pathlib import Path

STOP_WORDS = set(Path('data/stop_words_FULL.txt').open('r').read().splitlines())

class TopicExtractor():
    """Extract a topic from some text usign some method. The topic is a collection of nasari vectors.
    """
    def __init__(self, nasari):
        self._nasari = nasari
    
    def get_topic(self, text):
        title_tokens = bow_model(text, stopwords=STOP_WORDS)
        context_vectors = self._nasari.build_context(title_tokens)
        
        return context_vectors

class TitleExtractor(TopicExtractor):

    def __init__(self, lexical_resource):
        super().__init__(lexical_resource)

    def get_topic(self, document):
        """[summary]

        Args:
            text ([type]): [description]
            Document (data_manager.Document): named tuple to represent a semi-structured text document
        """
        #title_tokens = bow_model(document.title, stopwords=STOP_WORDS)
        #context_vectors = self._lex_resource.build_context(title_tokens)
        
        return super().get_topic(document.title)


def bow_model(text, stopwords=None):
    """Build a bag of word (BOW) model for a given sentence, 
    removing puntcuation marks and optionally stopwords. 

    Args:
        sentence (string): sentence from which build BOW.
        stopwords (set, optional): set of stopwords to remove from BOW. Defaults to None.

    Returns:
        set: bag of word.
    """
    bow = set()
    punct = {'.',',', ';', '(', ')', '{', '}', 
             '[', ']', "’", '‘',  '“', '”',':', '?', '!'} 
    lemmatizer = nltk.WordNetLemmatizer()

    for sentence in nltk.sent_tokenize(text): # sentence granularity tokenization
        sentence = sentence.lower()
        sentence_bow = set(nltk.word_tokenize(sentence)) # word granularity tokenization
        sentence_bow = sentence_bow.difference(punct)

        if stopwords:
            sentence_bow = sentence_bow.difference(stopwords)
        bow.update(sentence_bow)

    return set(lemmatizer.lemmatize(token) for token in bow) # lemmatization
