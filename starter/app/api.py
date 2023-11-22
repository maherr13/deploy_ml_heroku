"""
Author: Ahmed Maher
Date: Nov, 2023
This script hold the fastapi
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Body
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class FeatureInfo(str, Enum):
    age = "age"
    workclass = "workclass"
    fnlgt = "fnlgt"
    education = "education"
    education_num = "education_num"
    marital_status = "marital_status"
    occupation = "occupation"
    relationship = "relationship"
    race = "race"
    sex = "sex"
    captial_gain = "capital_gain"
    captial_loss = "capital_loss"
    hours_per_week = "hours_per_week"
    native_country = "native_country"


class Person(BaseModel):
    age: int
    workclass: Optional[str] = None
    fnlgt: int
    education: Optional[str] = None
    education_num: int
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Optional[str] = None

if "DYNO" in os.environ and os.path.isdir("../.dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r ../.dvc ../.apt/usr/lib/dvc")

app = FastAPI(
    title="Census",
    description="Deploying a ML Model FastAPI",
)

example_dir = "samples.yaml"
model_path = "md_lg.pkl"
model = joblib.load(model_path)
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
async def predict(person: Person = Body(..., examples=examples['post_examples'])):

    person = person.dict()
    features = np.array([person[f]
                        for f in examples['features_info'].keys()]).reshape(1, -1)
    df = pd.DataFrame(features, columns=examples['features_info'].keys())

    pred_label = int(model.predict(df))
    pred_probs = float(model.predict_proba(df)[:, 1])
    pred = '>50k' if pred_label == 1 else '<=50k'

    return {'label': pred_label, 'prob': pred_probs, 'salary': pred}