"""
Author: Ahmed Maher
Date: Nov, 2023
This script holds the conftest data used with pytest module
"""
import os
import pytest
import pandas as pd
import great_expectations as ge
from sklearn.model_selection import train_test_split


@pytest.fixture(scope='session')
def data():
    """
    Data loaded from csv file used for tests

    Returns:
        df (ge.DataFrame): Data loaded from csv file
    """
    if not os.path.exists('data/clean.csv'):
        pytest.fail(f"Data not found at path")
    df = pd.read_csv('data/clean.csv')
    df = ge.from_pandas(df)

    return df
