from collections import defaultdict, Counter

class BaseSmoother():
    def __init__(self, probs_dict=None, unknow_prob=1e-16):
        self.probs_dict = probs_dict
        self.unk_prob = unknow_prob

    def add_probabilities(self, probs_dict):
        self.probs_dict = probs_dict

    def get(self, key_from, key_to):
        return self.probs_dict[key_from].get(key_to, self.unk_prob)


class NounSmoother(BaseSmoother):
    def __init__(self, probs_dict=None):
        self.probs_dict = probs_dict
        
    def get(self, key_from, key_to):
        prob = 1.0 if key_from == "NOUN" else 0.0

        return self.probs_dict[key_from].get(key_to, prob)


class NounVerbSmoother(BaseSmoother):
    def __init__(self, probs_dict=None):
        self.probs_dict = probs_dict

    def get(self, key_from, key_to):
        prob = 0.5 if key_from in ["NOUN","VERB"] else 0.0

        return self.probs_dict[key_from].get(key_to, prob)



class UniformSmoother(BaseSmoother):
    def __init__(self, tags_set, probs_dict=None):
        self.probs_dict = probs_dict
        self.tags_set_size = len(tags_set)

    def get(self, key_from, key_to):
        prob = 1 / self.tags_set_size

        return self.probs_dict[key_from].get(key_to, prob)


import re
class RuleBasedSmoother(BaseSmoother):
    def __init__(self, patterns, probs_dict=None):
        self.probs_dict = probs_dict
        self._regexs = [(re.compile(regexp), tag,) for regexp, tag in patterns]

    def get(self, key_from, key_to):
        predicted_tag = None
        
        for regexp, tag in self._regexs:
            if re.match(regexp, key_to):
                predicted_tag = tag
                break

        prob = 1.0 if key_from == predicted_tag else 0.0

        return self.probs_dict[key_from].get(key_to, prob)
