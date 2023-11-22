import json 
import requests

data = {
        'age': 38,
        'fnlgt': 15,
        'education_num': 1,
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 5
    }


r  = requests.post("http://127.0.0.1:8000/predict/", json = json.dumps(data))

print(r.json())