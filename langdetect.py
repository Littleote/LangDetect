import argparse
import random
import ast

import pandas as pd
from sklearn.model_selection import train_test_split

import utils
from classifiers import applyClassifier, CLASSIFIERS
from preprocess import preprocess, PREPROCESSORS

seed = 42
random.seed(seed)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input data in csv format",
        metavar="FILE",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--voc_size",
        help="Vocabulary size",
        metavar="SIZE",
        type=int,
    )
    parser.add_argument(
        "--by_lang",
        help="Get vocabulary by language",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--analyzer",
        help="Tokenization level: {word, char}",
        type=str,
        choices=["word", "char"],
    )
    parser.add_argument(
        "-n",
        "--ngram_range",
        help="N-gram size range of the tokens",
        metavar=("FROM", "TO"),
        type=int,
        nargs=2,
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        help="Preprocessing procedure to apply to the corpus",
        type=str.upper,
        choices=list(PREPROCESSORS.keys()),
    )
    parser.add_argument(
        "-c",
        "--classifier",
        help="Classifier to predict languages",
        type=str.upper,
        choices=list(CLASSIFIERS.keys()),
    )

    def named_parameter(arg: str) -> dict:
        name, value_expr = arg.split(sep="=", maxsplit=1)
        value = ast.literal_eval(value_expr)
        return {name: value}

    parser.add_argument(
        "--args",
        help="Classifier arguments",
        type=named_parameter,
        metavar="NAME=VALUE",
        nargs="*",
        default=dict(),
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    kwargs = {k: v for arg in args.args for k, v in arg.items()}
    raw = pd.read_csv(args.input)

    # Languages
    languages = set(raw["language"])
    print("========")
    print("Languages", languages)
    print("========")

    # Split Train and Test sets
    X = raw["Text"]
    y = raw["language"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    print("========")
    print("Split sizes:")
    print("Train:", len(X_train))
    print("Test:", len(X_test))
    print("========")
    # Preprocess text (Word granularity only)
    if args.analyzer == "word":
        X_train, y_train = preprocess(X_train, y_train, use=args.preprocess)
        X_test, y_test = preprocess(X_test, y_test, use=args.preprocess)

    print(X_test)
    # Compute text features

    features, X_train_raw, X_test_raw = utils.compute_features(
        X_train,
        X_test,
        analyzer=args.analyzer,
        max_features=args.voc_size,
        ngram_range=args.ngram_range,
        languages=y_train if args.by_lang else None,
    )

    print("========")
    print("Number of tokens in the vocabulary:", len(features))
    print(
        "Coverage: ",
        utils.compute_coverage(
            features,
            X_test,
            analyzer=args.analyzer,
        ),
    )
    print("========")

    # Apply Classifier
    X_train, X_test = utils.normalizeData(X_train_raw, X_test_raw)
    y_predict = applyClassifier(args.classifier, X_train, y_train, X_test, **kwargs)

    print("========")
    print("Prediction Results:")
    utils.plot_F_Scores(y_test, y_predict)
    print("========")

    utils.plot_Confusion_Matrix(y_test, y_predict, "Greens")

    # Plot PCA
    print("========")
    print("PCA and Explained Variance:")
    utils.plotPCA(X_train, X_test, y_test, languages)
    print("========")
