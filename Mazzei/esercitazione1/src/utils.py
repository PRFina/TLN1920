from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

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


def evaluate(model, dataset_tokens, dataset_tags, labels):
    """Helper function to evaluate a model on a given dataset.

    Args:
        model (Any): one of the model in pos_tagging module.
        dataset_tokens (list[list[str]]): list of sentences, where each sentence is token sequence 
        dataset_tags (list[list[str]]): list of sentences, where each sentence is tag sequence
        labels (list[str]): tags name in the tagset

    Returns:
        float, np.ndarray: accuaracy and multiclass confusion matrix.
    """
    all_predictions = []
    all_true_tags = []

    for sentence, true_tags in zip(dataset_tokens[:2500], dataset_tags[:2500]):
        predicted_tags = model.predict(sentence)

        for (token, predicted), true in zip(predicted_tags, true_tags):
            all_predictions.append(predicted)
            all_true_tags.append(true)

    accuracy = accuracy_score(all_true_tags, all_predictions)
    confusion_mat = confusion_matrix(all_true_tags, all_predictions, labels=labels)

    return accuracy, confusion_mat

def load_pattern_rules(filepath):
    patterns_rules = []
    with Path(filepath).open("r") as rules_file:
        for line in rules_file.readlines():
            pattern, POS_tag = line.split(",")
            patterns_rules.append((pattern, POS_tag.strip("\n")))
    
    return patterns_rules