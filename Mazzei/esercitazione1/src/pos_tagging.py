import pandas as pd
import numpy as np

class ViterbiMatrix():

    def __init__(self, tag_set, tokens):
        self.pos_tags = sorted(tag_set)
        self.tokens = tokens
        self.matrix = pd.DataFrame(index=sorted(tag_set), columns=tokens, dtype=np.float64)
    
    def assign(self, pos, token_idx, value):
        self.matrix.loc[pos].iloc[token_idx] = value

    def get_prefix_probs(self, token_idx):
        return self.matrix.iloc(axis=1)[token_idx-1]