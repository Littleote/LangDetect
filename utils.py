import numpy as np
import pandas as pd
import scipy
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import normalize


def compute_features(
    X_train, X_test, analyzer="char", max_features=None, ngram_range=None
):
    """
    Task: Compute a matrix of token counts given a corpus.
          This matrix represents the frecuency any pair of tokens appears
          together in a sentence.

    Input: X_train -> Train sentences
           X_test -> Test sentences
           analyzer -> Granularity used to process the sentence
                      Values: {word, char}
           tokenizer -> Callable function to apply to the sentences before compute.

    Output: unigramFeatures: Cout matrix
            X_unigram_train_raw: Features computed for the Train sentences
            X_unigram_test_raw: Features computed for the Test sentences
    """

    ngram_range = (1, 1) if ngram_range is None else tuple(ngram_range)

    unigramVectorizer = CountVectorizer(
        analyzer=analyzer, max_features=max_features, ngram_range=ngram_range
    )

    X_unigram_train_raw = unigramVectorizer.fit_transform(X_train)
    X_unigram_test_raw = unigramVectorizer.transform(X_test)
    unigramFeatures = unigramVectorizer.get_feature_names_out()
    return unigramFeatures, X_unigram_train_raw, X_unigram_test_raw


def compute_coverage(features, split, analyzer="char"):
    """
    Task: Compute the proportion of a corpus that is represented by
          the vocabulary. All non covered tokens will be considered as unknown
          by the classifier.

    Input: features -> Count matrix
           split -> Set of sentence
           analyzer -> Granularity level {'word', 'char'}

    Output: proportion of covered tokens
    """
    total = 0.0
    found = 0.0
    for sent in split:
        # The following may be affected by your preprocess function. Modify accordingly
        sent = sent.split(" ") if analyzer == "word" else list(sent)
        total += len(sent)
        for token in sent:
            if token in features:
                found += 1.0
    return found / total


# Utils for conversion of different sources into numpy array
def toNumpyArray(data):
    """
    Task: Cast different types into numpy.ndarray
    Input: data ->  ArrayLike object
    Output: numpy.ndarray object
    """
    data_type = type(data)
    if data_type == np.ndarray:
        return data
    elif data_type == list:
        return np.array(data_type)
    elif data_type == scipy.sparse.csr.csr_matrix:
        return data.toarray()
    print(data_type)
    return None


def normalizeData(train, test):
    """
    Task: Normalize data to train classifiers. This process prevents errors
          due to features with different scale

    Input: train -> Train features
           test -> Test features

    Output: train_result -> Normalized train features
            test_result -> Normalized test features
    """
    train_result = normalize(train, norm="l2", axis=1, copy=True, return_norm=False)
    test_result = normalize(test, norm="l2", axis=1, copy=True, return_norm=False)
    return train_result, test_result


def plot_F_Scores(y_test, y_predict):
    """
    Task: Compute the F1 score of a set of predictions given
          its reference

    Input: y_test: Reference labels
           y_predict: Predicted labels

    Output: Print F1 score
    """
    f1_micro = f1_score(y_test, y_predict, average="micro")
    f1_macro = f1_score(y_test, y_predict, average="macro")
    f1_weighted = f1_score(y_test, y_predict, average="weighted")
    print(
        "F1: {} (micro), {} (macro), {} (weighted)".format(
            f1_micro, f1_macro, f1_weighted
        )
    )


def plot_Confusion_Matrix(y_test, y_predict, color="Blues"):
    """
    Task: Given a set of reference and predicted labels plot its confussion matrix

    Input: y_test ->  Reference labels
           y_predict -> Predicted labels
           color -> [Optional] Color used for the plot

    Ouput: Confussion Matrix plot
    """
    allLabels = list(set(list(y_test) + list(y_predict)))
    allLabels.sort()
    confusionMatrix = confusion_matrix(y_test, y_predict, labels=allLabels)
    unqiueLabel = np.unique(allLabels)
    df_cm = pd.DataFrame(confusionMatrix, columns=unqiueLabel, index=unqiueLabel)
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"
    sn.set_theme(font_scale=0.8)  # for label size
    sn.set_theme(rc={"figure.figsize": (15, 15)})
    sn.heatmap(
        df_cm, cmap=color, annot=True, annot_kws={"size": 12}, fmt="g"
    )  # font size
    plt.show()


def plotPCA(x_train, x_test, y_test, langs):
    """
    Task: Given train features train a PCA dimensionality reduction
          (2 dimensions) and plot the test set according to its labels.

    Input: x_train -> Train features
           x_test -> Test features
           y_test -> Test labels
           langs -> Set of language labels

    Output: Print the amount of variance explained by the 2 first principal components.
            Plot PCA results by language

    """
    pca = PCA(n_components=2)
    pca.fit(toNumpyArray(x_train))
    pca_test = pca.transform(toNumpyArray(x_test))
    print("Variance explained by PCA:", pca.explained_variance_ratio_)
    y_test_list = np.asarray(y_test.tolist())
    for lang in langs:
        pca_x = np.asarray([i[0] for i in pca_test])[y_test_list == lang]
        pca_y = np.asarray([i[1] for i in pca_test])[y_test_list == lang]
        plt.scatter(pca_x, pca_y, label=lang)
    plt.legend(loc="upper left")
    plt.show()
