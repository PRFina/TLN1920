from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd


class ViterbiMatrix():

    def __init__(self, tag_set, tokens):
        self.pos_tags = sorted(tag_set)
        self.tokens = tokens
        self.matrix = pd.DataFrame(index=sorted(tag_set), columns=tokens, dtype=np.float64)
    
    def assign(self, pos, token_idx, value):
        self.matrix.loc[pos].iloc[token_idx] = value

    def get_prefix_probs(self, token_idx):
        return self.matrix.iloc(axis=1)[token_idx-1]




class HMMPosTagger():
    START_TOKEN = "START"
    END_TOKEN = "END"
    ZERO_PROB = 1e-16

    def __init__(self):
        self.emission_probs = {}
        self.transition_probs = {}
        self.pos_tags = None # to discover after first pass
        self.viterbi_mat = None # will be instantiated in predict step
        pass


    def fit(self, X, y):

        pos_counts = Counter()
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)

        for sentence_tokens, sentence_tags in zip(X,y):

            # emission counts
            for token, pos_tag in zip(sentence_tokens, sentence_tags):
                emission_counts[token][pos_tag] += 1
                pos_counts[pos_tag] += 1
            
            # add once for each sentence
            pos_counts[HMMPosTagger.START_TOKEN] += 1
            pos_counts[HMMPosTagger.END_TOKEN] += 1
            
            # transtions counts
            curr_tags = sentence_tags + [HMMPosTagger.END_TOKEN]
            prev_tags = [HMMPosTagger.START_TOKEN] + sentence_tags
            for prev_tag, curr_tag in zip(prev_tags, curr_tags):
                transition_counts[prev_tag][curr_tag] += 1

        self.pos_tags = sorted(pos_counts.keys())
        import math

        #normalize = lambda qnt, total: math.log(qnt/(total))
        normalize = lambda qnt, total: qnt/(total)

        for token, counter in  emission_counts.items():
            self.emission_probs[token] = {token_pos: normalize(counter[token_pos], pos_counts[token_pos]) for token_pos in counter} 

        for pos, counter in transition_counts.items():
            self.transition_probs[pos] = {next_pos: normalize(counter[next_pos], pos_counts[next_pos]) for next_pos in counter}


    def predict(self, tokens, with_viterbi_matrix=False):

        pos_tags = [pos for pos in self.pos_tags if pos not in [HMMPosTagger.START_TOKEN, HMMPosTagger.END_TOKEN]]
        viterbi_mat = ViterbiMatrix(pos_tags, tokens)
        backpointers = defaultdict(list)

        # init
        for pos in viterbi_mat.pos_tags:
            emission_prob = self.emission_probs[tokens[0]].get(pos, HMMPosTagger.ZERO_PROB)
            viterbi_mat.assign(pos, 0, self.transition_probs[HMMPosTagger.START_TOKEN][pos] * emission_prob)

        # recursion
        for token_idx, token in enumerate(tokens[1:], start=1):
            for pos in viterbi_mat.pos_tags:
                emission_prob = self.emission_probs[token].get(pos, HMMPosTagger.ZERO_PROB)

                viterbi_prefix = viterbi_mat.get_prefix_probs(token_idx)
                transition_prefix = [self.transition_probs.get(prefix_tag, 0).get(pos, 0) for prefix_tag in viterbi_mat.pos_tags]

                viterbi_prob = (viterbi_prefix * transition_prefix).max() * emission_prob
                viterbi_mat.assign(pos, token_idx, viterbi_prob)
                
                backpointers[token_idx-1].append((pos, (viterbi_prefix * transition_prefix).idxmax()))

        # finalize
        end_token_idx = len(tokens)
        viterbi_prefix = viterbi_mat.get_prefix_probs(end_token_idx)
        transition_prefix = [self.transition_probs.get(prefix_tag, 0).get(HMMPosTagger.END_TOKEN, 0) for prefix_tag in viterbi_mat.pos_tags]

        backpointers[end_token_idx-1].append((HMMPosTagger.END_TOKEN, (viterbi_prefix * transition_prefix).idxmax()))

        # reconstruct backwardly the viterbi path
        predicted_tags = HMMPosTagger._reconstruct_path(tokens, backpointers)
        
        if with_viterbi_matrix:
            return predicted_tags, viterbi_mat.matrix
        else:
            return predicted_tags


    def _reconstruct_path(tokens, backpointers):
        predicted_tags = deque() # used for efficient insertion
        next_pos = None
        for token_idx in reversed(backpointers):

            if backpointers[token_idx][0][0] == HMMPosTagger.END_TOKEN:
                next_pos = backpointers[token_idx][0][1]
                predicted_tags.appendleft((tokens[token_idx], next_pos))
                continue

            for options in backpointers[token_idx]:
                if options[0] == next_pos:
                    next_pos = options[1]
                    predicted_tags.appendleft((tokens[token_idx], next_pos))
                    break

        return list(predicted_tags)


class DummyMajorityTagger():
    
    def __init__(self):
        self.emission_counts = defaultdict(Counter)
        self.pos_tags = None


    def fit(self, X, y):
        tags_set = set()
        for sentence_tokens, sentence_tags in zip(X,y):
            # emission counts
            for token, pos_tag in zip(sentence_tokens, sentence_tags):
                self.emission_counts[token][pos_tag] += 1
                tags_set.add(pos_tag)

        self.pos_tags = sorted(tags_set)

    def predict(self, tokens):
        predicted_tags = []
        for token in tokens:
            if token in self.emission_counts:
                most_common_tag = self.emission_counts[token].most_common()[0][0]
            else:
                most_common_tag = "NOUN"
        
            predicted_tags.append((token, most_common_tag))

        return predicted_tags
