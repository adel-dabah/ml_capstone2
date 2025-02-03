import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
url ="https://f0dhzqffpk.execute-api.eu-north-1.amazonaws.com/test_stage/predict"
data={'url' : 'https://github.com/adel-dabah/ml_capstone2/blob/main/dataset/Testing/pituitary/Te-piTr_0003.jpg?raw=true'}

result= requests.post(url, json=data).json()
print(result)