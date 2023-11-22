"""
Author: Ahmed Maher
Date: Nov, 2023
This script used for testing api
"""

import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def assert_response(response, expected_status, expected_method):
    assert response.status_code == expected_status
    assert response.request.method == expected_method


def test_greetings():
    response = client.get('/')
    assert_response(response, HTTPStatus.OK, "GET")
    assert response.json() == "Hello PPL"


@pytest.mark.parametrize('test_input, expected', [
    ('age', "Age of the person - numerical - int"),
    ('fnlwgt', 'MORE INFO NEEDED - numerical - int'),
    ('race', 'Race of the person - nominal categorical - str')
])
def test_feature_info(test_input: str, expected: str):
    response = client.get(f'/feature_info/{test_input}')
    assert_response(response, HTTPStatus.OK, "GET")
    assert response.json() == expected


def test_predict():
    data = {
        'age': 38,
        'workclass' : "state-gov",
        'education' : "bachelors",
        'maritalstatus' : "never-married",
        'occupation' : "adm-clerical",
        'relationship' : "not-in-family",
        'race' : "white",
        'sex' : "male",
        'nativecountry' : "united-states",
        'fnlwgt': 15,
        'educationnum': 1,
        'capitalgain': 0,
        'capitalloss': 0,
        'hoursperweek': 5,

    }
    response = client.post("/predict/", json=data)
    assert_response(response, HTTPStatus.OK, "POST")
    assert 0 <= response.json()['label'] <= 1
    assert response.json()['salary'] in ['>50k', '<=50k']


def test_missing_feature_predict():
    data = {"age": 0}
    response = client.post("/predict/", json=data)
    assert_response(response, HTTPStatus.UNPROCESSABLE_ENTITY, "POST")
    assert response.json()["detail"][0]["type"] == "value_error.missing"
