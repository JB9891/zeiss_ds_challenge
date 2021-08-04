import mlflow
from matplotlib import pyplot as plt
from src.data_processing import *


def get_logged_model(path_logged_model):
    """Loads logged model from mlflow runs

    :param path_logged_model: Path to logged model
    :type path_logged_model: str
    :return: Logged scikitlearn model
    :rtype: GridSearchCV
    """
    logged_model = path_logged_model

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model


def create_coeff_plot(model):
    coefs = model._model_impl.best_estimator_._final_estimator.coef_[0]
    features, _, _ = get_features()
    for i in range(len(coefs)):
        print(f"Feature {features[i]}: {coefs[i]}")

    # Plot the scores
    plt.bar(features, coefs)
    plt.xticks(rotation=90)
    plt.show()
    features, _, _ = get_features()

