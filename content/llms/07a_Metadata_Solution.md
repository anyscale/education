# Metadata generation and querying lab

This lab is an opportunity to familiarize yourself with
* Ray datasets
* `map_batches`
* generating metadata and providing it to a vector store
* using metadata to improve query results

If you're new to LLMs and Ray applications, focus on the core activities. If you've worked with LLMs and/or Ray before, you may have time to try the advanced activity.

## Core activities

Throughout this lab, we're going to work with additional metadata for each of our documents. The metadata we'll add here is simple: it's just the length of the document. But we'll see that having (and using) even trivial metadata like this allows us to improve our search results.

1. Copy and modify the code that uses Ray datasets and `.map_batches` to generate embeddings. Add to the ouput of the existing processing operation a column that contains the length of each document. *Hint: it will be another key-value pair in the dictionary representing the batch-processing output*
1. Set the actor pool to a fixed size of 4 instead of the autoscaling version we used before (just to explore the `map_batches` API further)
1. Using the ChromaDB docs and our existing code, generate a new Chroma collection that includes metadata for each doc. *Hint: the metadata will be supplied in a list alongside the docs and IDs. Each metadata record is a Python dictionary.*
1. Modify the calls to `collection.query` to handle a `where` condition that filters against the metadata. *Hint: since we have length metadata, we can query for shorter or longer documents*
1. Find a query where having and using the metadata makes a difference in the results -- ideally, producing better results.

## Advanced activity

In this activity, we're not trying to product any new output functionality.

But instead of modifying the existing actor class that generates the embeddings to also generate metadata, we'll leave that code as-is.

1. Create a function (instead of a class) for transforming the batches of data. *Hint: the function signature will look like this: `add_metadata(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]`*
1. We're going to use the original embedding generator first, and then apply this add_metadata transformation to the data batches after they emerge from the previous `map_batches` step (with the new schema including doc, id, and embedding)
1. We'll apply this second transformation via another call to `map_batches` but the call will be simpler than the previous one, since we don't have actors and Ray can handle scaling tasks on its own. We also don't need to worry about GPUs for this operation or specifying batch size.
1. Collect the output via `to_numpy_refs` and then `ray.get` one of those chunks of data, inspect it, and verify it has the same strucure as the actor-based implementation

## Core activity solution


```python
import uuid

import chromadb
import numpy as np
import ray
from InstructorEmbedding import INSTRUCTOR
```


```python
paras_ds = ray.data.read_text("/mnt/cluster_storage/around.txt", parallelism=4)
```


```python
class DocEmbedderWithMetadata:
    def __init__(self):
        self._model = INSTRUCTOR('hkunlp/instructor-large')

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        inputs = batch['text']
        embeddings = self._model.encode(inputs, device='cuda:0')
        ids = np.array([uuid.uuid1().hex for i in inputs])
        lengths = np.array([len(i) for i in inputs])
        return { 'doc' : inputs, 'vec' : embeddings, 'id' : ids, 'length' : lengths }
```


```python
vecs = paras_ds.map_batches(DocEmbedderWithMetadata, compute=ray.data.ActorPoolStrategy(size=4), num_gpus=0.25, batch_size=64)
```


```python
numpy_refs = vecs.to_numpy_refs()
```


```python
dicts = ray.get(numpy_refs)

vecs = np.vstack([d['vec'] for d in dicts])
ids = np.hstack([d['id'] for d in dicts])
docs = np.hstack([d['doc'] for d in dicts])
metadatas = sum( [ [{'length' : int(length) } for length in d['length']] for d in dicts ], [])
```


```python
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="metadata_lab")
```


```python
collection.upsert(
    embeddings=vecs.tolist(),
    documents=docs.tolist(),
    ids=ids.tolist(),
    metadatas=metadatas
)
```


```python
model = INSTRUCTOR('hkunlp/instructor-large')
```


```python
utah_query_vec = model.encode("Describe the body of water in Utah").tolist()
```


```python
def results_with_and_without_length(query_vec, length):
    where_filter =  { "length": { "$gt" : length  } }
    results_without_length = collection.query(
        query_embeddings=[query_vec],
        n_results=3
    )
    results_with_length = collection.query(
        query_embeddings=[query_vec],
        n_results=3,
        where=where_filter
    )
    return (results_without_length, results_with_length)
```

Adding this this specific metadata filter to this query does not make a difference...


```python
results_with_and_without_length(utah_query_vec, 200)
```

Adding a metadata filter with this query __does__ make a difference and improves the results:


```python
bank_query = model.encode('bank robbery details').tolist()
```


```python
results_with_and_without_length(bank_query, 200)
```

## Advanced activity solution


```python
class DocEmbedder:
    def __init__(self):
        self._model = INSTRUCTOR('hkunlp/instructor-large')

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        inputs = batch['text']
        embeddings = self._model.encode(inputs, device='cuda:0')
        ids = np.array([uuid.uuid1().hex for i in range(len(inputs))])
        return { 'doc' : inputs, 'vec' : embeddings, 'id' : ids }
```


```python
def add_metadata(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    lengths = np.array([len(i) for i in batch['doc']])
    batch['length'] = lengths
    return batch
```


```python
vecs = paras_ds \
        .map_batches(DocEmbedder, compute=ray.data.ActorPoolStrategy(size=4), num_gpus=0.25, batch_size=64) \
        .map_batches(add_metadata)
```


```python
numpy_refs = vecs.to_numpy_refs()
```


```python
ray.get(numpy_refs[0])
```
