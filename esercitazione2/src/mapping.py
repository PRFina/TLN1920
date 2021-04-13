
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
    """ Context Builder to extract context from a FrameNet Frame.
        A FrameNet frame have different slots, the context can be built from:

        * Frame definition.
        * Frame Elements (FE) definition.
        * Lexical Units (LU) definition.
    """
    def __init__(self) -> None:
        super().__init__()

    def get_context(self, frame, slot_value, slot_type):
        """ Build the context from a specific slot of the frame, 3 slot types are supported:

            * FrameNetSlotType.NAME: build the context from the frame definition
            * FrameNetSlotType.FE: build the context from a specific FE of the frame
            * FrameNetSlotType.FE: build the context from a specific LU of the frame

        Args:
            frame (ntlk.corpus.framenet.Frame): a Frame containing a slot_type with slot_value
            slot_value (str): value of the slot from which context is extracted.
            slot_type (FrameNetSlotType): type of the slot.

        Raises:
            ValueError: if slot_type arg is not a FrameNetSlotType enum value.

        Returns:
            [set of str]: the context of the given frame slot 
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
    """ Context Builder to extract context from a Wordnet Synset.
        There are multiple sources where context could be extracted:

        * synset gloss definition.
        * synset example sentences.
        * same as above but using synset hypernyms and hyponyms relations.
        
        Warning: this class use only hypernyms and hyponyms gloss definitions. Empirically, I noticed that
        taking in considerations also example sentences may produce too general context
        given a distorted view of the synset sense. 
    """
    
    def __init__(self) -> None:
        super().__init__()

    def get_context(self, synset):
        """ Build the context from a given sysnet.
        
        Args:
            synset (ntlk.corpus.wordnet.Synset): synset from which context is extracted

        Returns:
            [set of str]: the context of the given synset 
        """
        ctx = self._get_context(synset) # get synset examples sentences and gloss
        
        # get synset hypernyms context from gloss definitions
        for hypernym in synset.hypernyms():
            ctx.update(self._get_context(hypernym, examples=False))

        # get synset hyponyms context from gloss definitions
        for hyponym in synset.hyponyms():
            ctx.update(self._get_context(hyponym, examples=False))

        return ctx
        
        
    
    def _get_context(self, synset, examples=True):
        """Build bag of word based context from synset gloss definition
        and example sentences. 

        This method is a convenience one, the main method that should be called
        is the homonymous one without "_" in front of.

        Args:
            synset (ntlk.corpus.wordnet.Synset): synset from which context is extracted
            examples (bool, optional): if include in the context also example sentences. Defaults to True.

        Returns:
            [set of str]: the context for the fiven sysnet
        """
        ctx_s = set()
        bow = self.bow_model

        ctx_s.update(bow(synset.definition(), stopwords=STOP_WORDS))
        
        if examples:
            for example in synset.examples():
                ctx_s.update(bow(example, stopwords=STOP_WORDS)) 
        
        return ctx_s


class FNtoWNMapper():
    """ A mapper object to map a FrameNet frame slot to a wordnet sense (synset).

        The mapping algorithm is based on context overlapping measure:
        given a frame slot value w and a synset s with thei respective context ctx(w) and ctx(s),
        the returned wordnet sense is the one that maximize the overlapping:
                            |ctx(w) \cup \ctx(s)| + 1
    """

    def __init__(self, framenet_context_builder, wordnet_context_builder ) -> None:
        self._fn_ctx_builder = framenet_context_builder
        self._wn_ctx_builder = wordnet_context_builder

    def map(self, frame_name, slot_value, slot_type):
        """ Map a FrameNet frame to a wordnet sense.

        Args:
            frame_name (str): exact name of the frame (no regex expression)
            slot_value (str): value of the slot to map.
            slot_type (FrameNetSlotType): type of the slot to map

        Returns:
            nltk.corpus.wordnet.Synset: best wordnet sense for the given frame slot value.
        """
        synset_lemma = slot_value.split('.')[0] # get rid of POS (eg. existence.n)
        frame = fn.frame(frame_name)

        return self._best_sense(frame, slot_value, slot_type, synset_lemma)


    def _best_sense(self, frame, slot_value, slot_type, synset_lemma):
        """ Helper method that actually do the heavy work of 
            searching for the best sense for the given frame slot.

        Args:
            frame (nltk.corpus.framenet.Frame): FrameNet frame
            slot_value (str): value of the slot to which extract the context.
            slot_type (FrameNetSlotType): type of the slot to map
            synset_lemma (str): lemma for senses to search.

        Returns:
            nltk.corpus.wordnet.Synset: best wordnet sense for the given frame slot value.
        """
        
        fn_ctx = self._fn_ctx_builder.get_context(frame, slot_value, slot_type)
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
        """Given a Frame context and a synset context compute the context overlapping measure
        as the number of common tokens in both the bag-of-words.  

        Args:
            framenet_context (set of str): framenet context for a given slot
            wordnet_context (set of str): synset contex

        Returns:
            int: overlap measure
        """
        return len(framenet_context.intersection(wordnet_context)) + 1 # |ctx(w) + ctx(s)| + 1 
    