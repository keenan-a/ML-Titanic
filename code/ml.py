import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from random import randrange

def plot_survival_balance(train_df):
    num_dead = len(train_df[train_df["Survived"] == 0])
    num_alive = len(train_df[train_df["Survived"] == 1])

    plt.bar(x=["Survived", "Dead"], height=[num_dead, num_alive])
    plt.show()


def preprocess_data(df, task):
    """
    Preprocess the training dataframe to inlcude the interesting features we want to train with

    The classes so far are:
        - PClass
        - Fare
        - Sex
    Args:
        train_df (df): the training dataframe to be processed
    
    Return:
        Filtered df of interesting features
    """
    data_df = (pd.DataFrame([df["Fare"], df["Sex"], df["Pclass"], df["SibSp"], df["Parch"]])).T
    data_df = data_df.replace("male", 0)
    data_df = data_df.replace("female", 1)
    data_df = data_df.replace(np.NaN, randrange(5, 10))
    if task == "train":
        train_label_df = (pd.DataFrame([df["Survived"]])).T
        train_label_df = np.ravel(train_label_df)
    else:
        train_label_df = []
    return data_df, train_label_df

def rfc():
    return RandomForestClassifier()

def svc():
    return SVC(kernel="linear", C=0.025)

def gnb():
    return GaussianNB()

def main():
    train_df = pd.read_csv("../data/titanic/train.csv")
    test_df = pd.read_csv("../data/titanic/test.csv")

    train_data_df, train_label_df = preprocess_data(train_df, "train")
    test_data_df, _ = preprocess_data(test_df, "test")

    rf_classifier = svc()
    rf_classifier.fit(X=train_data_df, y=train_label_df)
    predictions = rf_classifier.predict(test_data_df)

    submit_df = (pd.DataFrame([test_df["PassengerId"], predictions])).T
    submit_df.columns = ["PassengerId", "Survived"]

    submit_df.to_csv("../data/titanic/submission.csv", index=False)

main()