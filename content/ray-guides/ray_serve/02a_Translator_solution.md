```python
import json
from typing import Dict

import ray
from ray import serve
from starlette.requests import Request
from transformers import pipeline
```


```python
@serve.deployment
class Translate:
    def __init__(self, model: str):
        self._model = model
        self._pipeline = None
    
    def get_response(self, user_input: str) -> str:
        if (self._pipeline is None):
            self._pipeline = pipeline(model=self._model)
        outputs = self._pipeline(user_input)
        return outputs
        
translate = Translate.bind(model='google/flan-t5-large')
```


```python
@serve.deployment(ray_actor_options={"runtime_env" : {"pip": ["lingua-language-detector==1.3.2"]}})
class LangDetect:
    def __init__(self):
        self._detector = None
        
    def get_response(self, user_input: str) -> str:
        from lingua import Language, LanguageDetectorBuilder
        
        if (self._detector is None):
            languages = [Language.ENGLISH, Language.ITALIAN]
            self._detector = LanguageDetectorBuilder.from_languages(*languages).build()
        
        output = self._detector.detect_language_of(user_input)
        if (output == Language.ENGLISH):
            return 'en'
        elif (output == Language.ITALIAN):
            return 'it'
        else:
            raise Exception('Unsupported language')
        
lang_detect = LangDetect.bind()
```


```python
@serve.deployment
class Endpoint:
    def __init__(self, lang_detect, translate):
        self._lang_detect = lang_detect
        self._translate = translate        

    async def __call__(self, request: Request) -> Dict:
        data = await request.json()
        data = json.loads(data)
        return {'response': await self.get_response(data['user_input']) }
    
    async def get_response(self, user_input: str):
        lang_obj_ref = await self._lang_detect.get_response.remote(user_input)
        lang = await lang_obj_ref

        if (lang == 'it'):
            prompt = "Translate to English: "      
        elif (lang == 'en'):
            prompt = "Translate to Italian: "
        else:
            raise Exception('Unsupported language')
        
        result = await self._translate.get_response.remote(prompt + user_input)       
            
        response = await result        
        return response
```


```python
endpoint = Endpoint.bind(lang_detect, translate)

endpoint_handle = serve.run(endpoint, name = 'translator')
```


```python
r = endpoint_handle.get_response.remote("I like playing tennis.")
```


```python
ray.get(r)
```


```python
ray.get(endpoint_handle.get_response.remote("Mi piace giocare a tennis"))
```


```python
serve.shutdown()
```
