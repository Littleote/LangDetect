import re

CUT_LEN = 2
CUT_THRESHOLD = 18


def cut(document):
    def split(match):
        cuts = len(match.group(0)) - CUT_LEN + 1
        return " ".join([match.group(0)[i : i + CUT_LEN] for i in range(cuts)])

    return re.sub(rf"\b\w{{{CUT_THRESHOLD},}}\b", split, document)


def word_limit(documents, labels):
    return documents.apply(cut), labels


def preprocess(documents, labels, *, use=None):
    """
    Task: Given a sentence apply all the required preprocessing steps
    to compute train our classifier, such as sentence splitting,
    tokenization or sentence splitting.

    Input: Sentence in string format
    Output: Preprocessed sentence either as a list or a string
    """
    fun = PREPROCESSORS.get(use, None)
    if fun is not None:
        return fun(documents, labels)
    return documents, labels


PREPROCESSORS = dict(
    WL=word_limit,
)
