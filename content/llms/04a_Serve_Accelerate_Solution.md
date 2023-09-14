```python
import json
from typing import AsyncGenerator
import requests
from fastapi import BackgroundTasks
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from ray import serve
```

Core deployment definition


```python
@serve.deployment(ray_actor_options={"num_gpus": 1})
class VLLMPredictDeployment:
    def __init__(self, **kwargs):
        # Refer to https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py for the full list of arguments.
        args = AsyncEngineArgs(**kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def stream_results(self, results_generator) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret = {"text": text_output}
            yield (json.dumps(ret) + "\n").encode("utf-8")
            num_returned += len(text_output)

    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def __call__(self, request: Request) -> Response:
        # The request should be a JSON object with the following fields: prompt, stream (True/False), kwargs for vLLM `SamplingParams`
        
        request_dict = await request.json()
        prompt = request_dict.pop("prompt")
        stream = request_dict.pop("stream", False)
        sampling_params = SamplingParams(**request_dict)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        if stream:
            background_tasks = BackgroundTasks()
            # Using background_taks to abort the the request
            # if the client disconnects.
            background_tasks.add_task(self.may_abort_request, request_id)
            return StreamingResponse(
                self.stream_results(results_generator), background=background_tasks
            )

        # Non-streaming case
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return Response(status_code=499)
            final_output = request_output

        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]
        ret = {"text": text_outputs}
        return Response(content=json.dumps(ret))
```

Our config for testing


```python
model='facebook/opt-125m'
download_dir='/mnt/local_storage'

prompt = 'What is your favorite place to visit in San Francisco?'
```

Start application on Serve


```python
deployment = VLLMPredictDeployment.bind(model=model, download_dir=download_dir)
serve.run(deployment, name='vllm')
```

Test and print output


```python
sample_input = {"prompt": prompt, "stream": True}
output = requests.post("http://localhost:8000/", json=sample_input)
for line in output.iter_lines():
    print(line.decode("utf-8"))
```

Run multiple requests asynchronously


```python
cities = ['Atlanta', 'Boston', 'Chicago', 'Vancouver', 'Montreal', 'Toronto', 'Frankfurt', 'Rome', 'Warsaw', 'Cairo', 'Dar Es Salaam', 'Gaborone']
prompts = [f'What is your favorite place to visit in {city}?' for city in cities]

def send(m):
    return requests.post("http://localhost:8000/", json={"prompt": m, "stream": True})

outputs = map(send, prompts)
```


```python
for output in outputs:
    for line in output.iter_lines():
        print(line.decode("utf-8"))
```

Change code to get 200 tokens in responses


```python
def send(m):
    return requests.post("http://localhost:8000/", json={"prompt": m, "stream": True, "max_tokens": 200})

outputs = map(send, prompts)
```


```python
for output in outputs:
    for line in output.iter_lines():
        print(line.decode("utf-8"))
```


```python
serve.shutdown()
```
