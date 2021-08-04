"""
This module contains functions for data preprocessing
"""

import pandas as pd


def preprocess_data(path_raw, features, label, path_secret):
    """Applies all data preprocessing steps

    :param path_raw: Path ro raw csv file
    :type path_raw: str
    :param features: List of manually selected features
    :type features: list
    :param label: Column name of label
    :type label: str
    :param path_secret: Path to place where secret test set is stored
    :type path_secret: str
    :return: Preprocessed training data
    :rtype: DataFrame
    """
    # Load raw data from csv file
    raw_data = load_raw(path_raw)

    # Clean raw data
    clean_data = clean(raw_data)

    # Impute data
    # imputed_data = impute(clean_data)
    imputed_data = clean_data
    # Split raw data into training and secret test data
    train, secret = split_train_secret(imputed_data)

    # Write secret test data to csv
    secret.to_csv(path_secret + "secret_test_data.csv", sep=',', index=False)

    # Filter relevant features and label of training data
    cols = features + [label]
    train = train[cols]

    return train

def load_raw(path):
    """Load raw data of challenge

    :param path: Path to csv file
    :type path: str
    :return: DataFrame of raw data
    :rtype: DataFrame
    """
    # read data
    df = pd.read_csv(path, sep=",")

    return df


def clean(data):
    """Accounts for the non numeric value in column q_OpeningHours

    :param data: Raw Data
    :type data: DataFrame
    :return: Cleaned Data
    :rtype: DataFrame
    """
    # Replace non numeric value with NaN
    data["q_OpeningHours"] = pd.to_numeric(
        data["q_OpeningHours"], errors="coerce"
    )

    return data


def impute(data):
    """Impute data

    :param data: Clean data
    :type data: DataFrame
    :return: Imputed data
    :rtype: DataFrame
    """
    # It is assumed that 0 values in Opening Hours are unrealistic do to the fact that some of the observations having
    # purchased something record a 0 value in Opening Hours

    # Impute 0 values in OpeningHours with median
    impute_val = data.loc[
        data["q_OpeningHours"] != 0, "q_OpeningHours"
    ].median()
    data.loc[data["q_OpeningHours"] == 0, "q_OpeningHours"] = impute_val

    return data


def split_train_secret(data):
    """Split raw data into training and secret test data

    :param data: Raw data of challenge
    :type data: DataFrame
    :return: Train and secret test data
    :rtype: DataFrame
    """
    train = data[data["b_in_kontakt_gewesen"] == 1].reset_index(drop=True)
    secret = data[data["b_in_kontakt_gewesen"] != 1].reset_index(drop=True)

    return train, secret


