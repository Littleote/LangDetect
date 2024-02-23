from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from utils import toNumpyArray


def applyNaiveBayes(X_train, y_train, X_test, **kwargs):
    """
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features
    Output: y_predict -> Predictions over the test set
    """
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = MultinomialNB(**kwargs)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


def applySupportVectorMachine(X_train, y_train, X_test, **kwargs):
    """
    Task: Given some features train an SVM classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features
    Output: y_predict -> Predictions over the test set
    """
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = SVC(**kwargs)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


def applyKNearestNeighbours(X_train, y_train, X_test, **kwargs):
    """
    Task: Given some features train a nearest neighbours classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features
    Output: y_predict -> Predictions over the test set
    """
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = KNeighborsClassifier(**kwargs)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


def applyClassifier(classifier, X_train, y_train, X_test, **kwargs):
    """
    Task: Given some features train some classifier
          and return its predictions over a test set
    Input; classifier -> Name of the classifier
           X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features
    Output: y_predict -> Predictions over the test set
    """
    fun = CLASSIFIERS.get(classifier, applyNaiveBayes)
    return fun(X_train, y_train, X_test, **kwargs)


CLASSIFIERS = dict(
    NB=applyNaiveBayes,
    SVM=applySupportVectorMachine,
    KNN=applyKNearestNeighbours,
)
