import requests

try:
    response = requests.post('http://localhost:5000/api/ai/decision')
    print("AI決策回應:")
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"請求錯誤: {e}")