```python
import ray
import torch
from ray import serve
from transformers import pipeline
```

# Autoscaling lab

In this lab we'll explore
* autoscaling
* "serverless" deployments
* replicas
* resources
* Ray dashboard

## Core activity

The main activity in this lab is to refactor basic Ray Serve chat deployment so that it supports autoscaling.

Using the "Review: scaling and performance" section below as a reference, adjust the chat deployment so that it has the following properties
* when not in use, it has no running replicas
* it starts off with no running replicas (since it's not in use "at launch time")
* it can scale to a maximum of 2 replicas

Look at the Ray dashboard and note the deployment and replica states.

By swapping in city names (e.g. 12 cities and 4 copies of each query), create a collection of 48 prompts and immediately call the model with all of them.

Look at the Ray dashboard and log messages and note the scaling behavior and reporting of this scaling.

## Bonus activity

See what happens if we try to scale to 3 replicas (knowing that we only have 2 x A10G GPUs in our cluster)

### Review: Scaling and performance

#### Replicas and autoscaling

Each deployment can have its own resource management and autoscaling configuration, with several options for scaling.

By default -- if nothing specified, as in our examples above -- the default is a single. We can specify a larger, constant number of replicas in the decorator:
```python
@serve.deployment(num_replicas=3)
```

For autoscaling, instead of `num_replicas`, we provide an `autoscaling_config` dictionary. With autoscaling, we can specify a minimum and maximum range for the number of replicas, the initial replica count, a load target, and more.

Here is example of extended configuration -- see https://docs.ray.io/en/releases-2.6.1/serve/scaling-and-resource-allocation.html#scaling-and-resource-allocation for more details:

```python
@serve.deployment(
    autoscaling_config={
        'min_replicas': 1,
        'initial_replicas': 2,
        'max_replicas': 5,
        'target_num_ongoing_requests_per_replica': 10,
    }
)
```

`min_replicas` can also be set to zero to create a "serverless" style design: in exchange for potentially slower startup, no actors (or their CPU/GPU resources) need to be permanently reserved.


```python
serve.shutdown()
```


```python
prompt = '''You are a helpful assistant.### User: What is your favorite place to visit in San Francisco? ### Assistant:'''

CHAT_MODEL = 'stabilityai/StableBeluga-7B'
```


```python
@serve.deployment(ray_actor_options={"num_gpus": 1.0})
class Chat:
    def __init__(self, model: str):
        self._model =  pipeline("text-generation", model=model, model_kwargs={
                                        'torch_dtype':torch.float16,
                                        'device_map':'auto',
                                        "cache_dir": "/mnt/local_storage"})
    
    def get_response(self, message: str) -> str:
        return self._model(message, max_length=200)

handle = serve.run(Chat.bind(model=CHAT_MODEL), name='chat')
```


```python
ref = handle.get_response.remote(prompt)
```


```python
ray.get(ref)
```


```python
serve.delete('chat')
```


```python
serve.shutdown()
```
