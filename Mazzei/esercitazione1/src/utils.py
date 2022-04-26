from pathlib import Path

def preprocess_data(dataset):
    """Preprocess input dataset.

    This utility function transform pyconll formatted data into a more uniform and general forma. 


    Args:
        dataset (pyconll.unit.conll.Conll): a pyconll dataset parsed with pyconll.read_from_file() functions 

    Returns:
        Tuple[List[list[str]], List[List[str]]]: a tuple of list. Each outer list items are sentences. 
                                                 Inner lists are tokens and pos tags of the sentence. 
    """
    tokens = []
    tags = []

    for sent in dataset:
        tokens.append([tok.lemma for tok in sent])
        tags.append([tok.upos for tok in sent])

    return tokens, tags


def load_pattern_rules(filepath):
    patterns_rules = []
    with Path(filepath).open("r") as rules_file:
        for line in rules_file.readlines():
            pattern, POS_tag = line.split(",")
            patterns_rules.append((pattern, POS_tag.strip("\n")))
    
    return patterns_rules