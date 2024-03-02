from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from utils import toNumpyArray


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
    cls = CLASSIFIERS.get(classifier, MultinomialNB)
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)

    clf = cls(**kwargs)
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


CLASSIFIERS = dict(
    NB=MultinomialNB,
    SVM=SVC,
    KNN=KNeighborsClassifier,
    RF=RandomForestClassifier,
)
