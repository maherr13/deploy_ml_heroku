
"""
Author: Ahmed Maher
Date: Nov, 2021
This script used for training, evaluting and saving the model
"""

# Script to train machine learning model.
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, evaluate_slices
# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import logging
import sys
import joblib
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Add code to load in the data.
logging.info("Loading Data ..")

path_metrics = '..\\metrics\\'
data = pd.read_csv('..\\data\\adult.data', header=None)

cols = ['age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary']

data.columns = cols

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

logging.info("Cleaning and Saving data ..")
data = data.replace('?', None)
data.dropna(inplace=True)
data.to_csv('..\\data\\clean.csv', index=False)

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary",
    training=False, encoder=encoder, lb=lb
)


logging.info("Training mdoels ..")
md_xgb = train_model(XGBClassifier(), X_train, y_train)
md_lg = train_model(LogisticRegression(C=0.5, max_iter=500), X_train, y_train)
md_rf = train_model(RandomForestClassifier(n_estimators=200), X_train, y_train)

logging.info("Saving results ..")
precision, recall, fbeta = compute_model_metrics(
    inference(md_xgb, X_test), y_test)
with open(path_metrics + 'eval_xgboot.txt', 'w') as file:
    file.write(r"percision: " + str(precision) + '\n')
    file.write(r"recall: " + str(recall) + '\n')
    file.write(r"fbeta: " + str(fbeta) + '\n')

precision, recall, fbeta = compute_model_metrics(
    inference(md_lg, X_test), y_test)
with open(path_metrics + 'eval_logistic_regression.txt', 'w') as file:
    file.write(r"percision: " + str(precision) + '\n')
    file.write(r"recall: " + str(recall) + '\n')
    file.write(r"fbeta: " + str(fbeta) + '\n')


precision, recall, fbeta = compute_model_metrics(
    inference(md_rf, X_test), y_test)
with open(path_metrics + 'eval_random_forest.txt', 'w') as file:
    file.write(r"percision: " + str(precision) + '\n')
    file.write(r"recall: " + str(recall) + '\n')
    file.write(r"fbeta: " + str(fbeta) + '\n')

logging.info("Evaluating slices and saving to file")

for col in ['sex', 'race']:
    folder = col+'_slice_train'
    slice_dir = f'..\\metrics\\{folder}\\'
    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)
    evaluate_slices(os.path.join(slice_dir,'slice_output.txt'),md_xgb, col, X_train, y_train, "train", train, lb)
    folder = col+'_slice_test'
    slice_dir = f'..\\metrics\\{folder}\\'
    if not os.path.exists(slice_dir):
        os.makedirs(slice_dir)
    evaluate_slices(os.path.join(slice_dir,'slice_output.txt'),md_xgb, col, X_test, y_test, "test", test, lb)

logging.info("Saving models ..")
joblib.dump(md_xgb, '..\\models\\model_xgboots.pkl')
joblib.dump(md_lg, '..\\models\\md_lg.pkl')
joblib.dump(md_rf, '..\\models\\model_random_forest.pkl')
