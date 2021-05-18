import numpy as np


class Hypothesis(object):

    def __init__(self, tokens, log_probs):
        self._tokens = tokens
        self._log_probs = log_probs

    def extend(self, token, log_prob):
        return Hypothesis(self._tokens + [token],
                          self._log_probs + [log_prob])

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

        # probabilità del singolo carattere del vocabolario, dal primo all'ultimo
        all_probs = model.predict(data)

        # per ogni cammino nella beam search
        for i, h in enumerate(hyps):
            # generiamo la distribuzione di probabilità

            # per ogni passo prende le probabilità
            probs = list(all_probs[i, step])
            probs = probs / np.sum(probs)  # normalizzazione

            # array degli indici delle posizioni dei caratteri più probabili per quel passo
            # esempio: alla prima iterazione avremo come primo carattere (più probabile)
            # il carattere in posizione 63, che è la "e"
            indexes = probs.argsort()[::-1]

            # scegliamo 2 * beam_size possibili espansioni dei cammini
            for j in range(hps.beam_size * 2):
                # TODO: Aggiungere qui!
                # Aggiungiamo a all_hyps l'estensione delle ipotesi con
                # il j-esimo indice e la sua probabilità
                temp = h.extend(indexes[j], probs[indexes[j]])
                all_hyps.append(temp)

        # teniamo solo beam_size cammini migliori
        hyps = []
        for h in sort_hyps(all_hyps):
            if h.last_token == vocab.char2id("</s>"):
                # if step >= hps.seq_len:
                # Confronta questa guardia con quella del while a riga 44
                # e troverai che è l'opposto, ovvero non entrerà mai in questo if,
                # per questo l'ho commentato
                results.append(h)
            else:
                hyps.append(h)

            if len(hyps) == hps.beam_size or len(results) == hps.beam_size:
                break

        if len(hyps) == 0:
            hyps = results

        # aggiorniamo data con i token migliori
        if step + 2 < hps.seq_len:
            tokens = np.matrix([h.tokens for h in hyps])
            data[:, :step + 2] = tokens

        step += 1

    if len(results) == 0:
        results = hyps

    hyps_sorted = sort_hyps(results)

    return hyps_sorted[0]


def sort_hyps(hyps):
    """
    Ordinare i cammini in ordine decrescente di probabilita' media. Per farlo,
    1. Basta richimare la funzione corretta tra avg_log_prob e log_prob (proprieties della classe top-level)
    2. per completarla, prendre spunto dalla versione greedy della makename
    """
    # TODO: Completare!
    toret = sorted(hyps, key=lambda x: x.avg_log_prob, reverse=True)
    return toret
