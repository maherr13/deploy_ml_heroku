"""
Author: Ahmed Maher
Date: Nov, 2023
This script used for testing types and ranges
"""

import great_expectations as ge


def test_columns_exist(data: ge.dataset.PandasDataset):
    expected_columns = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'educationnum',
        'maritalstatus',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capitalgain',
        'capitalloss',
        'hoursperweek',
        'nativecountry',
        'salary']
    for column in expected_columns:
        assert data.expect_column_to_exist(
            column)['success'], f"{column} does not exist"


def test_column_dtypes(data: ge.dataset.PandasDataset):
    expected_column_types = {
        'age': 'int64',
        'workclass': 'object',
        'fnlwgt': 'int64',
        'education': 'object',
        'educationnum': 'int64',
        'marital-status': 'object',
        'occupation': 'object',
        'relationship': 'object',
        'race': 'object',
        'sex': 'object',
        'capitalgain': 'int64',
        'capitalloss': 'int64',
        'hoursperweek': 'int64',
        'nativecountry': 'object',
        'salary': 'object'}
    for column, dtype in expected_column_types.items():
        assert data.expect_column_values_to_be_of_type(
            column, dtype)['success'], f"{column} should be of type {dtype}"


def test_education_num_column(data: ge.dataset.PandasDataset):
    assert data.expect_column_values_to_be_between('educationnum', 1, 17)[
        'success'], "education_num column includes unknown category"


def test_marital_status(data: ge.dataset.PandasDataset):
    expected_categories = [
        'never-married', 'married-civ-spouse', 'divorced',
        'married-spouse-absent', 'separated', 'married-af-spouse', 'widowed'
    ]
    assert data.expect_column_distinct_values_to_equal_set('maritalstatus', expected_categories)[
        'success'], "marital_status column includes unknown category"


def test_label_salary(data: ge.dataset.PandasDataset):
    expected_classes = ['<=50k', '>50k']
    assert data.expect_column_distinct_values_to_equal_set('salary', expected_classes)[
        'success'], "salary column includes more than two classes"


def test_hours_per_week_range(data: ge.dataset.PandasDataset):
    assert data.expect_column_values_to_be_between('hoursperweek', 1, 99)[
        'success'], "hours_per_week column is not within range of 1 and 99"
