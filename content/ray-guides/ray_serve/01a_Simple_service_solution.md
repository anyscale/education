# Lab solution: car payment calculator


```python
# car payment calculation (approximate)

def monthly_payment(cost, rate, years_of_loan):
    n = 365.25
    total_paid = cost * (((1 + ((rate/100.0)/n)) ** (n*years_of_loan)))
    per_month = total_paid / (12 * years_of_loan)
    return per_month
```


```python
monthly_payment(40000, 2, 4)
```


```python
from typing import Dict

import json
import ray
import requests
from ray import serve
from starlette.requests import Request
```


```python
@serve.deployment
class Chat:

    async def __call__(self, request: Request) -> Dict:
        data = await request.json()
        data = json.loads(data)
        return {"result": self.monthly_payment(data['cost'], data['rate'], data['years_of_loan']) }
    
    def monthly_payment(self, cost, rate, years_of_loan):
        n = 365.25
        total_paid = cost * (((1 + ((rate/100.0)/n)) ** (n*years_of_loan)))
        per_month = total_paid / (12 * years_of_loan)
        return per_month

handle = serve.run(Chat.bind(), name='car_payment')
```


```python
ray.get(handle.monthly_payment.remote(40_000, 7, 6))
```


```python
sample_json = '{ "cost" : 40000, "rate" : 7, "years_of_loan" : 6 }'
```


```python
requests.post("http://localhost:8000/", json = sample_json).json()
```


```python
serve.delete('car_payment')
```
