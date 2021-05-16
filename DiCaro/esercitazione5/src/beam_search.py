import numpy as np

class Hypothesis(object):

    def __init__(self, tokens, log_probs):

        self._tokens = tokens
        self._log_probs = log_probs

    def extend(self, token, log_prob):

        return Hypothesis(self._tokens+[token],
                          self._log_probs+[log_prob])

    @property
    def last_token(self):
        return self._tokens[-1]

    @property
    def log_prob(self):
        return sum(self._log_probs)

    @property
    def length(self):
        return len(self._tokens)

    @property
    def avg_log_prob(self):
        return self.log_prob / self.length

    @property
    def tokens(self):
        return self._tokens


def beam_search(model, vocab, hps):

    start_id = vocab.char2id('<s>')

    hyps = [Hypothesis([start_id], [0.0]) for _ in range(hps.beam_size)]

    step = 0
    results = []
    data = np.zeros((hps.beam_size, hps.seq_len))

    while step < hps.seq_len and len(results) < hps.beam_size:

        # prepariamo l'input per la rete neurale
        tokens = np.transpose(np.array([h.last_token for h in hyps]))
        data[:, step] = tokens


        all_hyps = []
        all_probs = model.predict(data)
        # per ogni cammino nella beam search
        for i, h in enumerate(hyps):
            # generiamo la distribuzione di probabilita'
            probs = list(all_probs[i,step])
            probs = probs / np.sum(probs)
            indexes = probs.argsort()[::-1]
            # scegliamo 2 * beam_size possibili espansioni dei cammini
            for j in range(hps.beam_size*2):
                # aggiungere ad all_hyps l'estensione delle ipotesi
                # con il j-esimo indice e la sua probabilita'
                pass

        # teniamo solo beam_size cammini migliori
        hyps = []
        for h in sort_hyps(all_hyps):
            if h.last_token == vocab.char2id("</s>"):
                if step >= hps.seq_len:
                    results.append(h)
            else:
                hyps.append(h)

            if len(hyps) == hps.beam_size or len(results) == hps.beam_size:
                break

        # aggiorniamo data con i token migliori
        if step+2 < hps.seq_len:
            tokens = np.matrix([h.tokens for h in hyps])
            data[:, :step+2] = tokens

        step += 1

    if len(results) == 0:
        results = hyps

    hyps_sorted = sort_hyps(results)

    return hyps_sorted[0]


def sort_hyps(hyps):
    # ordinare i cammini in ordine decrescente di probabilita' media
    pass
