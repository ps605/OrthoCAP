import requests

url = "https://orthocap.app.blufiori.co/studies/86352ad2-94e1-4fc2-aed2-b2f79b4f41f5/tlJOQueG8hc9m9Rn4P7lE9QNFSy2?patientName="
query_parameters = {"downloadformat": "csv"}

response = requests.get(url, params=query_parameters)
print(response.url)
print(response.ok)
print(response.status_code)

with open("data.zip", mode="wb") as file:
     file.write(response.content)