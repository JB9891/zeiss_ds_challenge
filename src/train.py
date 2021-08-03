# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit

import math
import statistics


from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

from data_processing import *

import logging


def metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred)
    prec = precision_score(actual, pred)
    rec = recall_score(actual, pred)
    auc = roc_auc_score(actual, pred)
    return acc, f1, prec, rec, auc


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Set a seed value
    seed_value = 23
    np.random.seed(seed_value)

    # Path to csv of raw data
    path_raw = "../data/CustomerData_LeadGenerator.csv"

    # Path to directory of storing secret test data
    path_secret = "../data/"

    # Manually selected features
    numeric_features = [
        "q_OpeningHours",
        "q_2017 Total Households",
        "q_2017 Purchasing Power: Per Capita",
        "q_2017 Medical Products: Per Capita",
    ]
    binary_features = [
        "b_specialisation_i",
        "b_specialisation_h",
        "b_specialisation_g",
        "b_specialisation_f",
        "b_specialisation_e",
        "b_specialisation_d",
        "b_specialisation_c",
        "b_specialisation_b",
        "b_specialisation_a",
        "b_specialisation_j",
    ]
    features = numeric_features + binary_features

    # Define label column
    label = "b_gekauft_gesamt"

    # Apply preprocessing steps inlc. reading, cleaning, imputation of data
    data = preprocess_data(path_raw, features, label, path_secret)

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    binary_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant'))])

    mixed_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features)])

    # Build classifier
    #classifier_pipeline = Pipeline(steps=[('mixed_transformer', mixed_transformer),
    #                                      ('classifier', DummyClassifier(strategy="stratified"))])

    classifier_pipeline = Pipeline(steps=[('mixed_transformer', mixed_transformer),
                                          ('classifier', svm.SVC(C=1))])

    # Score

    scores = cross_val_score(classifier_pipeline, data[features], data[label], cv=5, scoring='precision')
    scores_mean = statistics.median(scores)
    scores_std = statistics.stdev(scores)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(lr, "model")
