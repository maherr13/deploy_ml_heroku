import requests
import json



data = {
        'age': 38,
        'workclass' : "state-gov",
        'education' : "bachelors",
        'marital_status' : "never-married",
        'occupation' : "adm-clerical",
        'relationship' : "not-in-family",
        'race' : "white",
        'sex' : "male",
        'native-country' : "united-states",
        'fnlwgt': 15,
        'educationnum': 1,
        'capitalgain': 0,
        'capitalloss': 0,
        'hoursperweek': 5,

    }

r = requests.post("http://127.0.0.1:8000/predict/", data=json.dumps(data))

print(r.json())
