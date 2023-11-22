"""
Author: Ibrahim Sherif
Date: October, 2021
This script holds the conftest data used with pytest module
"""
import os
import pytest
import pandas as pd
import great_expectations as ge
from sklearn.model_selection import train_test_split
import config


@pytest.fixture(scope='session')
def data():
    """
    Data loaded from csv file used for tests

    Returns:
        df (ge.DataFrame): Data loaded from csv file
    """
    if not os.path.exists('data/clean.csv'):
        pytest.fail(f"Data not found at path")

    X_df['salary'] = y_df
    X_df['salary'] = X_df['salary'].map({1: '>50k', 0: '<=50k'})

    df = ge.from_pandas(X_df)

    return df
