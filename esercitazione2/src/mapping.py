
import nltk
from pathlib import Path
from enum import Enum
import numpy as np

from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn

STOP_WORDS = set(Path('data/stop_words_FULL.txt').open('r').read().splitlines())

class FrameNetSlotType(Enum):
    NAME = 0
    FE = 1
    LU = 2


class ContextBuilder:
    """ interface that wrap methods to extract a context from 
        a given object (eg. FrameNet frame, Wordnet synset, etc.).

        Here the context is represented with a bag-of-words model.
    """

    def __init__(self) -> None:
        self.bow_mode = None;
        self.context = None;


    def get_context(self):
        """Extract the context with some custom logic.
           When subclassing, this method must be implemented with
           input parameters that represents the object from which the context is extracted.   
           The returned object must be the bag-of-word of tokens extracted.
        """
        pass

    def bow_model(self, text, stopwords=None):
        """Build a bag-of-word (BOW) model for a given text, 
        removing punctuation marks and optionally stopwords.

        Each extracted token is also lemmatized.

        Args:
            text (str): sentence from which build BOW.
            stopwords (set, optional): set of stopwords to remove from BOW. Defaults to None.

        Returns:
            set: bag of word.
        """
        bow = set()
        punct = {'.',',', ';', '(', ')', '{', '}', 
                '[', ']', "’", '‘',  '“', '”',':', '?', '!',"'"} 
        lemmatizer = nltk.WordNetLemmatizer()

        for sentence in nltk.sent_tokenize(text): # sentence granularity tokenization
            sentence = sentence.lower()
            sentence_bow = set(nltk.word_tokenize(sentence)) # word granularity tokenization
            sentence_bow = sentence_bow.difference(punct)

            if stopwords:
                sentence_bow = sentence_bow.difference(stopwords)
            bow.update(sentence_bow)

        return set(lemmatizer.lemmatize(token) for token in bow) # lemmatization

class FrameNetContext(ContextBuilder):
    """Context Builder to extract context from a FrameNet Frame.
        A FrameNet frame have different slots, the context can be built from:

        * Frame definition.
        * Frame Elements (FE) definition.
        * Lexical Units (LU) definition.

    Args:
        ContextBuilder ([type]): ContextBuilder Superclass.
    """
    def __init__(self) -> None:
        super().__init__()

    def get_context(self, frame, slot_value, slot_type):
        """Build the context from a specific slot of the frame. 3 slot type are supported:

            * FrameNetSlotType.NAME: build the context from the frame definition
            * FrameNetSlotType.FE: build the context from a specific FE of the frame
            * FrameNetSlotType.FE: build the context from a specific LU of the frame

        Args:
            frame (ntlk.corpus.framenet.Frame): a Frame object istance
            slot_value (str): value of the slot indicateb by slot_type
            slot_type (FrameNetSlotType): type of the slot.

        Raises:
            ValueError: if slot_type arg is not a FrameNetSlotType enum value.

        Returns:
            [set]: context of the given frame slot 
        """
        
        definition = ""

        if slot_type  == FrameNetSlotType.NAME:
            definition = frame.definition
    
        elif slot_type == FrameNetSlotType.FE:
            definition = frame['FE'][slot_value].definition
        
        elif slot_type == FrameNetSlotType.LU:
            definition = frame['lexUnit'][slot_value].definition
        else:
            raise ValueError("element_type is allowed only to get mapping.FrameNetElement enum values")
        
        return self.bow_model(definition, stopwords=STOP_WORDS)


class WordNetContext(ContextBuilder):
    def __init__(self) -> None:
        super().__init__()

    def get_context(self, synset):
        """Get context for the given sysnet

        Context is a bag of word built from:
        synset gloss definition
        synset examples
        all hypernyms and hyponyms synset gloss definitions --> Why? because empirically,
        examples may produce too general context given a distorted view of the synset sense. 

        Args:
            synset ([type]): [description]

        Returns:
            [type]: [description]
        """
        ctx = self._get_context(synset)
        
        for hypernym in synset.hypernyms():
            ctx.update(self._get_context(hypernym, examples=False))

        for hyponym in synset.hyponyms():
            ctx.update(self._get_context(hyponym, examples=False))

        return ctx
        
        
    
    def _get_context(self, synset, examples=True):
        """Build bag of word based context from synset gloss definition
        and example sentences. 

        This method is a convenience one, the main method that should be called
        is the homonymous one eithout "_" in front of.

        Args:
            synset ([type]): [description]
            examples (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        ctx_s = set()
        bow = self.bow_model

        ctx_s.update(bow(synset.definition(), stopwords=STOP_WORDS))
        
        if examples:
            for example in synset.examples():
                ctx_s.update(bow(example, stopwords=STOP_WORDS)) 
        
        return ctx_s


class Mapper():

    def __init__(self, framenet_context_builder, wordnet_context_builder ) -> None:
        self._fn_ctx_builder = framenet_context_builder
        self._wn_ctx_builder = wordnet_context_builder

    def map(self, frame_name, frame_element, element_type):
        synset_lemma = frame_element.split('.')[0] # get rid of POS 
        frame = fn.frame(frame_name)

        return self._best_sense(frame, frame_element, element_type, synset_lemma)


    def _best_sense(self, frame, frame_element, element_type, synset_lemma):
        
        fn_ctx = self._fn_ctx_builder.get_context(frame, frame_element, element_type)
        scores = []
        senses_id = []
        
        for synset in wn.synsets(synset_lemma): 
            wn_ctx = self._wn_ctx_builder.get_context(synset)
            scores.append(self._score_sense(fn_ctx, wn_ctx))
            senses_id.append(synset.name())

        idx_max = np.argmax(scores) if len(scores) > 0 else None
        best_sense = senses_id[idx_max] if idx_max is not None else None
        
        return best_sense 

    def _score_sense(self, framenet_context, wordnet_context):
        return len(framenet_context.intersection(wordnet_context)) + 1 # |ctx(w) + ctx(s)| + 1 
    