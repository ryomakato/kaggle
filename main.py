import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.preprocessing import StandardScaler

import titanic_model as titanic


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns={0: '欠損数', 1: '%'})
    return kesson_table_ren_columns


def preprocessing(df):
    df['Title'] = df.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
    Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Lady": "Royalty",
        "Mme": "Mrs",
        "Ms": "Mrs",
        "Mrs": "Mrs",
        "Mlle": "Miss",
        "Miss": "Miss",
        "Mr": "Mr",
        "Master": "Master"
    }

    # we map each title to correct category
    df['Title'] = df.Title.map(Title_Dictionary)

    df.loc[df.Age.isnull(), 'Age'] = df.groupby(['Sex', 'Pclass', 'Title']).Age.transform('median')

    # Do the same to test dataset
    interval = (0, 5, 12, 18, 25, 35, 60, 120)

    # same as the other df train
    cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']

    # same that we used above in df train
    df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)

    # Filling the NA's with -0.5
    df.Fare = df.Fare.fillna(-0.5)

    # intervals to categorize
    quant = (-1, 0, 8, 15, 31, 600)

    # Labels without input values
    label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

    # doing the cut in fare and puting in a new column
    df["Fare_cat"] = pd.cut(df.Fare, quant, labels=label_quants)

    del df["Fare"]
    del df["Ticket"]
    del df["Age"]
    del df["Cabin"]
    del df["Name"]

    df["Embarked"] = df["Embarked"].fillna('S')

    df["FSize"] = df["Parch"] + df["SibSp"] + 1

    del df["SibSp"]
    del df["Parch"]

    df = pd.get_dummies(df, columns=["Sex", "Embarked", "Age_cat", "Fare_cat", "Title"],
                              prefix=["Sex", "Emb", "Age", "Fare", "Prefix"], drop_first=True)

    return df

def train_model(data, labels):

    model = titanic.Model(num_classes=1)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss=tf.losses.mean_squared_error,
                  metrics=['accuracy'])
    model.fit(data, labels, batch_size=32, epochs=20)
    model.summary()

    return model

def main():

    # Importing train dataset
    df_train = pd.read_csv("./input/train.csv")
    # Importing test dataset
    df_test = pd.read_csv("./input/test.csv")

    # preprocessing
    df_train = preprocessing(df_train)
    df_test = preprocessing(df_test)

    # arengementing data
    train = df_train.drop(["Survived", "PassengerId"], axis=1)
    train_ = df_train["Survived"]
    test_ = df_test.drop(["PassengerId"], axis=1)
    X_train = train.values
    y_train = train_.values
    X_test = test_.values

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # train
    model = train_model(X_train, y_train)

    results = model.predict(X_test)
    results = np.where(results >= 0.5, 1, 0)

    # PassengerIdを取得
    PassengerId = np.array(df_test["PassengerId"]).astype(int)
    # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    my_solution = pd.DataFrame(results, PassengerId, columns=["Survived"])
    # my_tree_one.csvとして書き出し
    my_solution.to_csv("prediction.csv", index_label=["PassengerId"])


if __name__ == "__main__":
    main()
