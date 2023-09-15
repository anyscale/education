```python
import ray
from ray import serve
import json
from starlette.requests import Request
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```

# Deployment Graph API

What is the Deployment Graph API?
* A declarative (graph) composition pattern
* which lets us separate the flow of calls from the logic inside our services.

Why might we want to use the Deployment Graph (DAG) API to separate flow from logic?

* It may be valuable to add a layer of indirection – or abstraction – so that we can more easily create and compose reusable services
* The DAG API lets us use similar patterns across the Ray platform (e.g., Ray Workflow)
    * We can learn one general pattern for graphs and use that intuition in multiple places in our Ray applications
* Although we compose one DAG, we retain the key Ray Serve features of granular autoscaling and resource allocation

Let’s reproduce our chat service flow using the Deployment Graph API

<div class="alert alert-block alert-info">
    
__Roadmap to multilingual chat with deployment graphs__

1. Learn core concepts with a basic linear graph to begin a French-only chat
1. Demonstrate split/combine graph pattern to fully enable French chat
1. Add conditional flow for English/French chat
    
</div>

## Getting started with deployment graphs

As a first step, to keep things simple, let’s assume for a moment that we are always interacting with the service in French. 

{{< image src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Ray_Serve/deployment_graph_simple.png" >}}


```python
from ray.serve.dag import InputNode
from ray.serve.drivers import DAGDriver
```

`InputNode` is a special type of graph node, defined by Ray Serve, which represents values supplied to our service endpoint. 

We can only have one `InputNode` but we can get access to multiple parameters from that node using a Python context manager.


```python
with InputNode() as inp:
    user_input = inp[0]
    history = inp[1]
```

Here is a minimal, linear pipeline that allows us to begin a chat in French.

We build up the graph step by step, `bind`ing each deployment to its dependencies.


```python
@serve.deployment(ray_actor_options={"runtime_env" : { "pip": ["sentencepiece==0.1.99"]} })
class Translate:
    def __init__(self, task: str, model: str):
        self._task = task
        self._model = model
        self._pipeline = None
    
    def get_response(self, user_input: str) -> str:
        if (self._pipeline is None):
            self._pipeline = pipeline(task=self._task, model=self._model)
        outputs = self._pipeline(user_input)
        response = outputs[0]['translation_text']
        return response
        
translate_en_fr = Translate.bind(task='translation_en_to_fr', model='t5-small')
translate_fr_en = Translate.bind(task='translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
```


```python
@serve.deployment(ray_actor_options={'num_gpus': 0.5})
class Chat:
    def __init__(self, model: str):
        # configure stateful elements of our service such as loading a model
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model =  AutoModelForSeq2SeqLM.from_pretrained(model).to(0)

    async def __call__(self, request: Request) -> dict:
        # path to handle HTTP requests
        data = await request.json()
        data = json.loads(data)
        # after decoding the payload, we delegate to get_response for logic
        return {'response': self.get_response(data['user_input'], data['history']) }
    
    def get_response(self, user_input: str, history: list[str]) -> str:
        # this method receives calls directly (from Python) or from __call__ (from HTTP)
        history.append(user_input)
        # the history is client-side state and will be a list of raw strings;
        # for the default config of the model and tokenizer, history should be joined with '</s><s>'
        inputs = self._tokenizer('</s><s>'.join(history), return_tensors='pt').to(0)
        reply_ids = self._model.generate(**inputs, max_new_tokens=500)
        response = self._tokenizer.batch_decode(reply_ids.cpu(), skip_special_tokens=True)[0]
        return response
    
chat = Chat.bind(model='facebook/blenderbot-400M-distill')
```


```python
user_input_en = translate_fr_en.get_response.bind(user_input)    # French->English translator depends on the user input text
chat_response = chat.get_response.bind(user_input_en, history)   # the chat deployment requires the English user input and the history
output = translate_en_fr.get_response.bind(chat_response)        # English->French translator depends on the English chat output
serve_dag = DAGDriver.bind(output)                               # the graph returns the output from the English->French translator

handle = serve.run(serve_dag, name='basic_linear')
```

We start the application by calling `serve.run()` on the DAGDriver, a Ray Serve component which routes HTTP requests through your call graph.


```python
ray.get(handle.predict.remote('Mes amis sont cool mais ils mangent trop de glucides.', []))
```


```python
serve.delete('basic_linear')
```

How can we continue the chat?

We need to supply English history ... but we only have French responses so far.

We can use the pattern of adding a __combine node__ to our graph in order to merge the 3 elements we need to output (English chat message, English chat response, and French chat response).

Combining multiple values is a common requirement -- e.g., in collecting values from a model ensemble.

{{< image src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Ray_Serve/ensemble.png" >}}


```python
@serve.deployment
def combine(user_input_en:str, chat_response_en: str, chat_response_fr:str)->str:
    return chat_response_fr + '|' + user_input_en + '|' + chat_response_en
```

The combine node here implemented here is a very simple deployment: it's built from a single function definition instead of a class.


```python
translate_en_fr = Translate.bind(task='translation_en_to_fr', model='t5-small')
translate_fr_en = Translate.bind(task='translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
chat = Chat.bind(model='facebook/blenderbot-400M-distill')
```

Event though the definitions of the `Translate` and `Chat` deployments have not changed, we call `.bind()` again to create new DAG nodes since we're composing a new DAG.


```python
with InputNode() as inp:
    user_input = inp[0]
    history = inp[1]
    user_input_en = translate_fr_en.get_response.bind(user_input)
    chat_response_en = chat.get_response.bind(user_input_en, history)
    chat_response_fr = translate_en_fr.get_response.bind(chat_response_en)

# We route the user input, the English chat response, and the French chat response into the combine node
output = combine.bind(user_input_en, chat_response_en, chat_response_fr)

# and we serve the output of the combine node
serve_dag = DAGDriver.bind(output)

handle = serve.run(serve_dag, name='enhanced_linear')
```


```python
ray.get(handle.predict.remote('Mes amis sont cool mais ils mangent trop de glucides.', []))
```


```python
serve.delete('enhanced_linear')
```

Using this pattern, we are getting everything back that we would need to offer a conversation service with the chatbot ... but only in French!

## Adding Conditional Flow

Our real chatbot is a bit more complex. It has a conditional flow where we invoke the translation service only when the user is *not* interacting in English.

We can add the remaining elements of our service and the basic API changes will be fairly minimal. But there is one aspect that requires us to do a little bit of thinking and employ a new pattern.

### Static Graphs and Conditional Control Flow

The graph we define with the DAG API is static – it’s created ahead of time. 

In the first DAG demo, we were always invoking the same sequence of services, so the static character of the graph might not have been obvious… but now we’re focusing on it so you can see where things might get a bit more complicated.

To implement branching flow control with the DAG API, we’ll use a special pattern so that the same graph always runs … but certain nodes (in our case, translator nodes) behave differently based on data they receive.

{{< image src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Ray_Serve/deployment_graph_complex.png" >}}


```python
@serve.deployment(ray_actor_options={"runtime_env" : { "pip": ["sentencepiece==0.1.99"]} })
class Translate:
    def __init__(self, task: str, model: str):
        self._task = task
        self._model = model
        self._pipeline = None
    
    def get_response(self, user_input:str, user_lang:str) -> str:
        if (user_lang == 'en'):
            return user_input # no-op
        
        if (self._pipeline is None):
            self._pipeline = pipeline(task=self._task, model=self._model)
            
        outputs = self._pipeline(user_input)
        response = outputs[0]['translation_text']
        return response        
        

translate_en_fr = Translate.bind(task='translation_en_to_fr', model='t5-small')
translate_fr_en = Translate.bind(task='translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
```

The if-else control flow inside `get_response()` calls the transation logic only when the user is *not* using English.


```python
@serve.deployment(ray_actor_options={"runtime_env" : {"pip": ["lingua-language-detector==1.3.2"]}})
class LangDetect:
    def __init__(self):
        self._detector = None
        
    def get_response(self, user_input: str) -> str:
        from lingua import Language, LanguageDetectorBuilder
        
        if (self._detector is None):
            languages = [Language.ENGLISH, Language.FRENCH]
            self._detector = LanguageDetectorBuilder.from_languages(*languages).build()
        
        output = self._detector.detect_language_of(user_input)
        if (output == Language.ENGLISH):
            return 'en'
        else:
            return 'fr'
        
lang_detect = LangDetect.bind()
```


```python
chat = Chat.bind(model='facebook/blenderbot-400M-distill')

with InputNode() as inp:
    user_input = inp[0]
    history = inp[1]
    user_lang = lang_detect.get_response.bind(user_input)
    user_input_en = translate_fr_en.get_response.bind(user_input, user_lang)
    chat_response_en = chat.get_response.bind(user_input_en, history)
    chat_response_fr = translate_en_fr.get_response.bind(chat_response_en, user_lang)
    output = combine.bind(user_input_en, chat_response_en, chat_response_fr)
    serve_dag = DAGDriver.bind(output)

handle = serve.run(serve_dag, name='full_chatbot')
```

In this code, the translation services are always part of the graph and participate in the data flow. So the graph is static, even though the translation behavior is dynamic.


```python
message = 'Mes amis sont cool mais ils mangent trop de glucides.'
history = []

response = ray.get(handle.predict.remote(message, history))

response.split('|')[0]
```


```python
history += response.split('|')[1:]
history
```


```python
ray.get(handle.predict.remote('Truly bread is delightful', history))
```


```python
serve.delete('full_chatbot')
```
