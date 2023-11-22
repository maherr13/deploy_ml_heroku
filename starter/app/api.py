"""
Author: Ahmed Maher
Date: Nov, 2023
This script hold the fastapi
"""

import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from enum import Enum
from typing import Optional
from pydantic import BaseModel


cat_features = [
    "workclass",
    "education",
    "maritalstatus",
    "occupation",
    "relationship",
    "race",
    "sex",
    "nativecountry",
]

def process_data(
        X,
        categorical_features=[],
        encoder=None):
    
    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    X_categorical = encoder.transform(X_categorical)

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X



class FeatureInfo(str, Enum):
    age = "age"
    workclass = "workclass"
    fnlwgt = "fnlwgt"
    education = "education"
    educationnum = "educationnum"
    maritalstatus = "maritalstatus"
    occupation = "occupation"
    relationship = "relationship"
    race = "race"
    sex = "sex"
    captialgain = "capitalgain"
    captialloss = "capitalloss"
    hoursperweek = "hoursperweek"
    nativecountry = "nativecountry"


class Person(BaseModel):
    age: int
    workclass: Optional[str] = None
    fnlwgt: int
    education: Optional[str] = None
    educationnum: int
    maritalstatus: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    capitalgain: int
    capitalloss: int
    hoursperweek: int
    nativecountry: Optional[str] = None

if "DYNO" in os.environ and os.path.isdir("../.dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r ../.dvc ../.apt/usr/lib/dvc")

app = FastAPI(
    title="Census",
    description="Deploying a ML Model FastAPI",
)

example_dir = "starter/app/samples.yaml"
model_path = "starter/models/md_lg.pkl"
encoder = "stater/models/encoder.pkl"
model = joblib.load(model_path)
encoder = joblib.load(encoder)
with open(example_dir) as fp:
    examples = yaml.safe_load(fp)


@app.get("/")
async def greetings():
    return "Hello PPL"


@app.get("/feature_info/{feature_name}")
async def feature_info(feature_name: FeatureInfo):

    info = examples['features_info'][feature_name]
    return info


@app.post("/predict/")
async def predict(person: Person):
    print(person)
    person = person.dict()
    features = np.array([person[f]
                        for f in examples['features_info'].keys()]).reshape(1, -1)
    df = pd.DataFrame(features, columns=examples['features_info'].keys())
    X = process_data(X=df, categorical_features=cat_features, encoder=encoder)
    pred_label = int(model.predict(X))
    pred = '>50k' if pred_label == 1 else '<=50k'

    return {'label': pred_label, 'salary': pred}
