# Tecnologie per il Linguaggio Naturale

La repository contiene le esercitazioni per il corso Tecnologie per il Linguaggio naturale.

La repository è così organizzata:
* Il corso è diviso in 3 moduli principali tenuti da differenti professori. Le 3 cartelle top-level si riferiscono a questi moduli.
* Ogni modulo prevede una serie variabli di esercitiazioni. Ogni esercitazione ha una directory associata **self contained**, ovvero tute le risorse richieste per l'esecuzione delle esercitazioni sono incluse nella directory e non ci sono dipendenze esterne.
* Viene seguita la seguente convenzione per la stuttura delle directories:
    * `data`: contiene tutte le risorse lessicali e non utilizzate dal codice (es. corpus, file di input con annotazioni, ecc.).
    * `output`: contiene tutti gli eventuali artifatti di putput prodotti dall'esecuzione del codice.
    * `src`: contiene le implementazione delle classi e funzioni richieste dai notebook. Solitamente si trova un file `data_manager.py` contenente classi e  helper function che effettuano il parsing e offrono un API di supporto per l'accesso alle risorse lessicali contenute in `data`.
    * `esercitazione<n>`.ipynb` notebook principale in cui veine descritta ed eseguita l'esercitazione richiesta.
