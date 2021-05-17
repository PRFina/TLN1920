
from collections import Counter

import nltk.stem
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn

from textblob import TextBlob
import src.word_sense_disambiguation as wsd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def generate_genus_candidates(concept_signature):
    # Step 1 pos tagging
    def_pos = {}
    for definition in concept_signature:
        pos = nltk.pos_tag(nltk.word_tokenize(definition))
        def_pos[definition] =  pos
    
    # Step 2 genus extraction
    ambiguous_genera = {}
    singularizer = nltk.stem.WordNetLemmatizer()
    
    for defn in def_pos:
        # extract nouns (and also singularize NNS plurals nouns)
        candidate_genera = list(map(lambda lemma_pos: singularizer.lemmatize(lemma_pos[0]), 
                            filter(lambda x: x[1] in ['NN','NNS'], def_pos[defn])))
        ambiguous_genera[defn] = [TextBlob(candidate).correct().raw for candidate in candidate_genera] # fix mispelling

    # Step 3 candidate genus identification trough WSD
    genus_candidates = Counter()
    # WSD for genus and add to a multiset to keep track of occurences in the definitions
    for defn in ambiguous_genera:
        for genus in ambiguous_genera[defn]:
            best_sense = lesk(wsd.bow_model(defn), genus, pos=wn.NOUN) 
            if best_sense: # avoid None
                genus_candidates.update([best_sense])

    # Step 4 ranking heuristic based on occurence frequencies
    genus_candidates_ranking = list(map(lambda rank: rank[0], genus_candidates.most_common()))

    return genus_candidates_ranking

def hyponyms_signatures(genus_synset, max_search_depth):
    genus_hyponyms = list(genus_synset.closure(lambda syn: syn.hyponyms(), depth=max_search_depth))
    # join synset def with example to augment contextual informations
    definitions = [" ".join([hyp.definition()] +
                            hyp.examples()) for hyp in genus_hyponyms]
    # add the genus itself ???? (just to avoid empty hyponyms set)
    return [genus_synset] + genus_hyponyms, [genus_synset.definition()] +  definitions



def compute_similarity_matrix(concept_signature, hyponyms_signatures):
    vectorizer = CountVectorizer(stop_words='english')
    
    # vectorize both concept and hyponyms definitions (signatures)
    concept_mat = vectorizer.fit_transform(concept_signature)
    hyponyms_mat = vectorizer.transform(hyponyms_signatures)
    #print(hyponyms_signatures)
    similarity_mat = cosine_similarity(concept_mat, hyponyms_mat)

    return similarity_mat

def find_best_sense(similarity_matrix, genus_hyponyms):
    avg_similarities = similarity_matrix.mean(axis=0)
    best_hyponym_idx = np.argmax(avg_similarities)
    return genus_hyponyms[best_hyponym_idx], avg_similarities[best_hyponym_idx]



def content_to_form(concept_signature, top_k, max_search_depth):

    # search for genus candidates candid
    genus_candidates = generate_genus_candidates(concept_signature)

    candidate_senses = []
    # search for candidate senses 
    for genus_candidate in genus_candidates:
        hyponyms, hyp_signatures = hyponyms_signatures(genus_candidate, max_search_depth) 
        sim_mat = compute_similarity_matrix(concept_signature, hyp_signatures)
        candidate_senses.append(find_best_sense(sim_mat, hyponyms))

    # create a ranking by semantic relatdness (similarity)
    candidate_senses.sort(key=lambda x:x[1], reverse=True)
    # take the first top_k in the ranking
    return candidate_senses[0:top_k]