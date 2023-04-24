import requests

url = "http://localhost:80/lemmatize"
text = "behaviour donation sparking"
response = requests.get(url, json={"text": text})
if response.status_code == 200:
    lemmas = response.json()
    print(lemmas)
else:
    print("Error:", response.text)