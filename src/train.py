"""
Module to train some classifiers for zeiss coding challenge.

The module trains a logistic regression regularized by lasso norm
and a random forest classifier. Some hyperparameters are tuned with a simple grid search
and selected within k-fold cross validation. Additionally a dummy classifier simply prediction the most frequent class
(b_gekauft_gesamt = 1).

Precision is used as scoring metric, because it is assumed, that the lead generator is a prioritization problem, becasue
it is only possible to contact several of all potential customers via e.g. cold calls.
Thus a small amount of false positives would be highly desirable to save resources.
"""

import os
import warnings
import sys

import pandas as pd
import numpy as np


from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import mlflow

import logging

from data_processing import *


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


if __name__ == "__main__":

    # Path to csv of raw data
    path_raw = "../data/CustomerData_LeadGenerator.csv"

    # Path to directory of storing secret test data
    path_secret = "../data/"

    # Get selected features
    features, numeric_features, binary_features = get_features()

    # Define label column
    label = "b_gekauft_gesamt"

    # Apply preprocessing steps incl. reading, cleaning, imputation of data
    train_test = preprocess_data(path_raw, features, label, path_secret)

    # Use whole training data for model selection with k-fold cross validation
    # Nested cross validation is not applied due to already small sample size
    X = train_test[features]
    y = train_test[label]

    # Define folds of cross validation
    cv = StratifiedKFold(n_splits=4)

    # Mixed transformer to standard scale numeric features and keep binary features as they are
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    binary_transformer = Pipeline(steps=[("imputer", SimpleImputer())])

    mixed_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", binary_transformer, binary_features),
        ]
    )

    # Helper function for cv of dummy classifier
    def cv_dummy(params):  # Dummy classifier is used as benchmark

        # Cross validation of dummy classifier
        dummy_search = GridSearchCV(
            estimator=DummyClassifier(),
            param_grid=params,
            cv=cv,
            scoring="precision",
            return_train_score=True,
        )

        # Fit dummy classifier
        dummy_search.fit(X, y)

        return dummy_search

    # Helper function for cv and grid_search of logistic regression
    # Logistic regression with lasso norm to reduce high dimensionality of feature space
    # Especially in the context of the small sample size
    def cv_log(params,):

        log_pipeline = Pipeline(
            steps=[
                ("mixed_transformer", mixed_transformer),
                (
                    "logistic",
                    LogisticRegression(
                        solver="saga", max_iter=10000, penalty="l1",
                    ),
                ),
            ]
        )

        # Cross validation of log regression
        log_search = GridSearchCV(
            estimator=log_pipeline,
            param_grid=log_params,
            cv=cv,
            scoring="precision",
            return_train_score=True,
        )

        log_search.fit(X, y)

        return log_search

    # Helper function for cv and grid_search of random forest
    # Random forest is used as model benchmark with higher complexity
    def cv_rf(params,):

        # Cross validation of random forest
        rf_search = GridSearchCV(
            estimator=RandomForestClassifier(n_estimators=20),
            param_grid=params,
            cv=cv,
            scoring="precision",
            return_train_score=True,
        )

        # Cross validate random forest classifier
        rf_search.fit(X, y)

        return rf_search

    # Define experiments and grid search parameters

    # Parameters for Dummy Classifier
    dummy_params = {"strategy": ["most_frequent"]}

    # Parameter for LogisticRegression_L1
    log_params = {
        # Allow strict regularization
        "logistic__C": [1e-3, 1e-1, 1e0, 1e1, 1e2],
        # Exclude strict regularization
        # "logistic__C": [1e0, 1e1, 1e2],
    }

    # Parameters for random forest
    rf_params = {
        "max_depth": [1, 2, 4, 8, 16, 32, None],
        "max_features": [2, 4, 8],
    }

    # All experiments to be tested
    experiments = ["dummy", "logistic_regression_L1", "random_forest"]

    # Run experiments
    # Precision is used as scoring factor, because it is assumed
    # that the lead generator is a prioritization problem.
    for experiment in experiments:

        # Set a seed value
        seed_value = 23
        np.random.seed(seed_value)

        # Start mflow run
        mlflow.start_run(run_name=experiment)

        # Fit models
        if experiment == "dummy":
            clf = cv_dummy(dummy_params)
        elif experiment == "logistic_regression_L1":
            clf = cv_log(log_params)
        elif experiment == "random_forest":
            clf = cv_rf(rf_params)
        else:
            logger.info(f"No experiment defined for {experiment}")
            mlflow.end_run()
            continue

        params = {
            "best_params": clf.best_params_,
            "n_cv_folds": clf.n_splits_,
            "cv_scorer": clf.scoring,
        }

        idx = clf.best_index_
        metrics = {
            "mean_test_score_best": clf.cv_results_["mean_test_score"][idx],
            "std_test_best": clf.cv_results_["std_test_score"][idx],
            "mean_train_score_best": clf.cv_results_["mean_train_score"][idx],
            "std_train_best": clf.cv_results_["std_train_score"][idx],
        }

        # Log metrics and parameters of experiments
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Log the sklearn model and register as version 1
        model_name = "sk-learn-zeiss_cc-" + experiment
        mlflow.sklearn.log_model(clf, model_name)

        # End mlflow run
        mlflow.end_run()
        logger.info(f"Experiments for " + experiment + " finished")
