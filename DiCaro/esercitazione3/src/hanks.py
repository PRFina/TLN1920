from collections import namedtuple, Counter

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

import spacy

# pos tags for verbs, see https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
VERB_POS_TAGS = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# stanford parser dependencies types
SUBJ_DEP_TYPES = ['nsubj', 'nsubjpass']
OBJ_DEP_TYPES = ['dobj', 'iobj']

# Step 1

def corpus_extraction(verb, corpus,  verb_pos_tags = VERB_POS_TAGS):
    """Extract all sentences from corpus in which the input verb occour.
    To search for the verb occurences a pos tagging and lemmatization is executed.

    Args:
        verb (str): verb to search for
        corpus (nltk.corpus): corpus from which sentences are extracted
        verb_pos_tags (list of str, optional): accepted POS tags for verb. Defaults to VERB_POS_TAGS.

    Returns:
        list of list of str: list of tokenized sentences.
    """
    lemmatizer = WordNetLemmatizer()
    sentences = corpus.sents()

    selected_sentences = []
    for sent in sentences:
        tags = dict(nltk.pos_tag(sent))
        for word in sent:
            if tags[word] in VERB_POS_TAGS: # extract only verbs
                word = lemmatizer.lemmatize(word, 'v')
                if word == verb: # extract only sentences with the given verb
                    selected_sentences.append(sent)

    return selected_sentences

# Step 2

def find_target_verbs(doc, verb, verb_pos_tags = VERB_POS_TAGS):
    """search and extract all verb tokens occurrences in the parsed sentence.

    Args:
        doc (spacy.Doc): a spacy document container that represent a single sentence
        verb (str): target
        verb_pos_tags (list of str, optional): allowed verb POS tags. Defaults to VERB_POS_TAGS.

    Returns:
        list of spacy.Token: all verb occurences as spacy tokens
    """
    targets = []
    for token in doc:
        # search for verbs POS first and also in the lemmatized form
        if token.tag_ in verb_pos_tags and (token.text == verb or 
                                            token.lemma_ == verb):
            targets.append(token)
    return targets


def get_hanks_verb(verb_token, dep_allowed_types=OBJ_DEP_TYPES + SUBJ_DEP_TYPES):
    """Build a named tuple that contains information about the verb arguments and fillers
       using the parsed syntactic dependencies.

    The verb_token represent the root of the parsing tree cointaing the syntactic relations.

    This function doesn't make any check on the verb valency but only on the syntactic relation
    type.
    Args:
        verb_token (spacy.Token): [description]
        dep_allowed_types (list of str, optional): types of syntactic relations to search. Defaults to OBJ_DEP_TYPES+SUBJ_DEP_TYPES.

    Returns:
        named tuple: a named tuple with the following fileds:
        verb: name of the verb
        nargs: number of verb arguments found
        slot_1,...,slot_nargs: syntactic relation types
        filler_1,...,filler_nargs: value of the argument filler
    """
    slot_values = []
    filler_values = []
    
    # loop trough verb syntactic dependencies subtree (children)
    for children in verb_token.children:
        if children.dep_ in dep_allowed_types:
            slot_values.append(children.dep_) # save info about the relation type
            filler_values.append(children.text) # take the actual filler value

    # dynamically create a named tuple based on the number of slot/fillers found
    slot_names = [f"slot{i}" for i,_ in enumerate(slot_values, start=1)]
    filler_names = [f"filler{i}" for i,_ in enumerate(filler_values, start=1)]
    
    field_names = ['verb', 'nargs'] + slot_names + filler_names
    field_values = [verb_token.text, len(slot_names)] + slot_values + filler_values

    HanksVerb = namedtuple("HanksVerb", field_names)

    return HanksVerb(*field_values)

def find_verb_fillers(sentences, verb, nlp_pipeline, valence=2):
    """extract all verb fillers occurences for a given valence.

    Args:
        sentences ([list of list of str]): list of tokenized sentences.
        verb (str): target verb
        nlp_pipeline (spacy.nlp): a spacy nlp pipeline instance
        valence (int, optional): number of verb arguments. Defaults to 2.

    Returns:
        list of tuples (sentence, HanksVerb)
    """

    fillers = []

    # this tokenization step is necessary for Spacy since the pipeline takes a string as input
    detokenizer = TreebankWordDetokenizer()
    joined_sents = [detokenizer.detokenize(sent) for sent in sentences]

    for sentence, joined_sent in zip(sentences, joined_sents):
        doc = nlp_pipeline(joined_sent)
        
        # find target verb occurences in the sentence
        target_verbs = find_target_verbs(doc, verb)
        
        # loop trough target verb occurences
        for target_verb in target_verbs:
            
            # retrieve verb fillers from syntactic dependencies
            hank_verb = get_hanks_verb(target_verb)
            # check for valence
            if hank_verb.nargs == valence:
                fillers.append((sentence, hank_verb))
    return fillers


# step 3: WSD & semantic clustering

def find_filler_senses(fillers, wsd_func):
    filler_senses = []
    
    for sentence, hanks_verb in fillers:
        filler1_sense = wsd_func(sentence, hanks_verb.filler1)
        filler2_sense = wsd_func(sentence, hanks_verb.filler2)

        filler_senses.append((filler1_sense, filler2_sense))
    
    return filler_senses

def semantic_clustering(filler_senses):
    semantic_types = Counter()

    for filler1_sense, filler2_sense in filler_senses:
        if filler1_sense is not None and filler2_sense is not None:
                # implicit clustering with wn supersenses
                semantic_type = (filler1_sense.lexname(), filler2_sense.lexname())
                # keep track of semantic_type occurrences
                semantic_types.update([semantic_type])
        else:
            semantic_types.update([(None, None)]) # just to keep track of invalid semantic type senses
        
    return semantic_types

# put everything together
def compute_hanks(verb, valence, corpus, wsd_func):
    # step 1: subcorpus extraction
    selected_sents = corpus_extraction(verb, corpus)

    # step 2: filler extraction
    nlp = spacy.load("en_core_web_md")
    fillers = find_verb_fillers(selected_sents, verb, nlp_pipeline=nlp, valence=valence)

    # step 3: filler senses WSD
    filler_senses = find_filler_senses(fillers, wsd_func)

    # step 4: semantic clustering
    semantic_types = semantic_clustering(filler_senses)

    return fillers, filler_senses, semantic_types