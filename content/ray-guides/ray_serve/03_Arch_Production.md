```python
from ray import serve
import requests, json
from starlette.requests import Request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
```

# Architecture / under-the-hood

## Ray cluster perspective: actors

In Ray, user code is executed by worker processes. These workers can run tasks (stateless functions) or actors (stateful class instances).

Ray Serve is built on actors, allowing deployments to collect expensive state once (such as loading a ML model) and to reuse it across many service requests.

Although you may never need to code any Ray tasks or actors yourself, your Ray Serve application has full access to those cluster capabilities and you may wish to use them to implement other functionality (e.g., service or operations that don't need to accept HTTP traffic). More information is at https://docs.ray.io/en/releases-2.6.1/ray-core/walkthrough.html

## Serve design

Under the hood, a few other actors are used to make up a serve instance.

* Controller: A global actor unique to each Serve instance is responsible for managing other actors. Serve API calls like creating or getting a deployment make remote calls to the Controller.

* HTTP Proxy: By default there is one HTTP proxy actor on the head node that accepts incoming requests, forwards them to replicas, and responds once they are completed. For scalability and high availability, you can also run a proxy on each node in the cluster via the location field of http_options.

* Deployment Replicas: Actors that execute the code in response to a request. Each replica processes requests from the HTTP proxy.

<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Ray_Serve/serve-system-architecture.png" width="70%" loading="lazy">

Incoming requests, once resolved to a particular deployment, are queued. The requests from the queue are assigned round-robin to available replicas as long as capacity is available. This design provides load balancing and elasticity. 

Capacity can be managed with the `max_concurrent_queries` parameter to the deployment decorator. This value defaults to 100 and represents the maximum number of queries that will be sent to a replica of this deployment without receiving a response. Each replica has its own queue to collect and smooth incoming request traffic.

# Production features: scaling, performance, and more

<div class="alert alert-block alert-info">
    
__Roadmap: Production features__
    
1. Replicas and autoscaling
1. Request batching
1. Fault tolerance
1. Serve CLI, in-place upgrades, multi-application support

</div>

## Replicas and autoscaling

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

### Autoscaling LLM chat

The LLM-based chat service is a good example for seeing autoscaling in action, because LLM inference is relative expensive so we can easily build up a queue of requests to the service. The autoscaler responds to the dynamics of queue sizes and will launch additional replicas.


```python
@serve.deployment(ray_actor_options={'num_gpus': 0.5}, autoscaling_config={ 'min_replicas': 1, 'max_replicas': 4 })
class Chat:
    def __init__(self, model: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model =  AutoModelForSeq2SeqLM.from_pretrained(model).to(0)

    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        data = json.loads(data)
        return {'response': self.get_response(data['user_input'], data['history']) }
    
    def get_response(self, user_input: str, history: list[str]) -> str:
        history.append(user_input)
        inputs = self._tokenizer('</s><s>'.join(history), return_tensors='pt').to(0)
        reply_ids = self._model.generate(**inputs, max_new_tokens=500)
        response = self._tokenizer.batch_decode(reply_ids.cpu(), skip_special_tokens=True)[0]
        return response
    
chat = Chat.bind(model='facebook/blenderbot-400M-distill')

handle = serve.run(chat, name='autoscale_chat')
```

We can generate a little load and look at the Ray Dashboard

What do we expect to see?

* Autoscaling of the Chat service up to 4 replicas
* Efficient use of fractional GPU resources
    * If our cluster has just 2 GPUs, we can run 4 replicase there


```python
def make_request(s):
    return requests.post("http://localhost:8000/", json = s).json()

sample = '{ "user_input" : "Hello there, chatbot!", "history":[] }'
make_request(sample)
```


```python
executor = ThreadPoolExecutor(max_workers=32)

results = executor.map(make_request, ['{ "user_input" : "Hello there, chatbot!", "history":[] }'] * 128)
```


```python
serve.delete('autoscale_chat')
```

### Request batching

Many services -- especially services that rely on neural net models -- can produce higher throughput on batches of data.

At the same time, most service interfaces or contracts are based on a single request-response.

Ray Serve enables us to meet both of those goals by automatically applying batching based on a specified batch size and batch timeout.


```python
@serve.deployment()
class Chat:
    def __init__(self):
        self._message = "Chatbot counts the batch size at "

    @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.01)
    async def handle_batch(self, request_batch):
        num_requests = len(request_batch)
        return [ {'response': self._message + str(num_requests) } ] * num_requests
    
    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        data = json.loads(data)
        return await self.handle_batch(data)
    
chat = Chat.bind()

handle = serve.run(chat, name='batch_chat')
```


```python
results = executor.map(make_request, ['{ "user_input" : "Hello there, chatbot!", "history":[] }'] * 100)
```


```python
batches = [int(resp['response'].split(' ')[-1]) for resp in results]
```


```python
plt.hist(batches)
```


```python
serve.delete('batch_chat')
```

### Fault tolerance

Serve provides some fault tolerance features out of the box

* Replica health-checking: by default, the Serve controller periodically health-checks each Serve deployment replica and restarts it on failure
  * __Built in__: does not require KubeRay
  * Support for ustom application-level health-checks, frequency, and timeout
  * If the health-check fails, the Serve controller logs the exception, kills the unhealthy replica(s), and restarts them

End-to-end fault tolerance by running Serve on top of KubeRay or Anyscale

* Worker node restart
* Head node restart
* Head node state recovery with Redis

While Ray can start/restart/scale worker processes, KubeRay and Anyscale provide the ability to recover nodes, provision additional nodes from a resource pool, cloud provider, etc.

### Additional production considerations and features

#### Web application capabilities

* FastAPI support
* WebSockets
* Streaming responsea

#### Serve CLI

For use in production, Serve includes a CLI with commands to deploy applications, check them, update them, and more


```python
! serve status
```


```python
handle = serve.run(chat, name='batch_chat')
```


```python
! ray status
```


```python
! serve status
```


```python
serve.shutdown()
```

#### In-place upgrades and multi-application support

While deployments can be reconfigured in-place and hot-redeployed, those updates will trigger an update of all deployments within the application.

In large, complex applications, you may want to share a single Ray cluster and make updates to individual components, but not redeploy the entire set of services. For those use cases, Ray Serve allows you do define multiple applications.

This collection of applications
* runs in the same Ray cluster
* can interact with each other and lookup other services by name
* can be upgraded independently
