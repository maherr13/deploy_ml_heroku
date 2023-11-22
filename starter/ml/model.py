
"""
Author: Ahmed Maher
Date: Nov, 2023
This script used for functions utlization for models
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score
import logging
import pandas as pd

# Optional: implement hyperparameter tuning.


def train_model(model, X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using
    precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    result = model.predict(X)
    return result


def slice_metrics(column, X, y_true, y_pred, lb):
    """
    Calculates metrics on a slice of data for a specific column

    Args:
        column (str): Column name representing a feature
        X (pandas dataframe): data features
        y_true (pandas series): data true labels
        y_pred (pandas series): data predicted labels

    Returns:
        pandas dataframe: Dataframe with metrics for each category
    """
    df = pd.concat([X[column].copy(), y_true], axis=1)
    df['salary_pred'] = y_pred

    metrics = []
    for categ in df[column].unique():
        subset_df = df[df[column] == categ]
        y = lb.transform(subset_df['salary'].values).ravel()
        prec, rec, f1 = compute_model_metrics(subset_df['salary_pred'], y)
        metrics.append([categ, prec, rec, f1])

    return pd.DataFrame(
        metrics,
        columns=[
            'Category',
            'Precision',
            'Recall',
            'F1'])


def evaluate_slices(output_file, model_pipe, column, X, y, split, data, lb):
    """
    Evaluating model on a slice of data for a specific column
    and data split and saving the results to a file

    Args:
        model_pipe (sklearn pipeline/model): sklearn model or pipeline
        column (str): Column name representing a feature
        X (pandas dataframe): data features
        y (pandas series): data labels
        split (str): train or test split

    Returns:
        None
    """
    logging.info(f"Evaluating {column} on slice of {split} data")

    y_pred = inference(model_pipe, X)
    slice_df = slice_metrics(column, data.drop(
        'salary', axis=1), data[['salary']], y_pred, lb)

    with open(output_file, "a") as file:
        print(f"Model evaluation on {column} slice of {split} data", file=file)
        print(slice_df.to_string(index=False), file=file)
        print("", file=file)
