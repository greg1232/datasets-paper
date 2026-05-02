import os
import requests
API_TOKEN = os.environ["HF_TOKEN"]
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://huggingface.co/api/datasets/temp-data-store/batch_0000/croissant"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
print(data)
import json
json.dump(data, open("./dataset.json", "w"))
