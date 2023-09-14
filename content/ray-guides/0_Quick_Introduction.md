# 0. Introduction to the Ray AI Libraries
---

<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Ray_AI_Libraries/Ray+AI+Libraries.png" width="70%" loading="lazy">

Built on top of Ray Core, the Ray AI Libraries inherit all the performance and scalability benefits offered by Core while providing a convenient abstraction layer for machine learning. These Python-first native libraries allow ML practitioners to distribute individual workloads, end-to-end applications, and build custom use cases in a unified framework.

The Ray AI Libraries bring together an ever-growing ecosystem of integrations with popular machine learning frameworks to create a common interface for development.

|<img src="https://technical-training-assets.s3.us-west-2.amazonaws.com/Introduction_to_Ray_AIR/e2e_air.png" width="100%" loading="lazy">|
|:-:|
|Ray AI Libraries enable end-to-end ML development and provides multiple options for integrating with other tools and libraries form the MLOps ecosystem.|

**Table of Contents**
1. [**Ray Data**](https://docs.ray.io/en/latest/data/dataset.html)  
    * Scalable, framework-agnostic data loading and transformation across training, tuning, and inference.
2. [**Ray Train**](https://docs.ray.io/en/latest/train/train.html)
    * Distributed multi-node and multi-core training with fault tolerance that integrates with popular machine learning libraries.
3. [**Ray Tune**](https://docs.ray.io/en/latest/tune/index.html)  
    * Scales hyperparameter tuning to optimize model performance.
4. [**Ray Serve**](https://docs.ray.io/en/latest/serve/index.html)  
    * Deploys a model or ensemble of models for online inference.

## Quick end-to-end example

|Ray AIR Component|NYC Taxi Use Case|
|:--|:--|
|Ray Data|Ingest and transform raw data; perform batch inference by mapping the checkpointed model to batches of data.|
|Ray Train|Use `Trainer` to scale XGBoost model training.|
|Ray Tune|Use `Tuner` for hyperparameter search.|
|Ray Serve|Deploy the model for online inference.|

For this classification task, you will apply a simple [XGBoost](https://xgboost.readthedocs.io/en/stable/) (a gradient boosted trees framework) model to the June 2021 [New York City Taxi & Limousine Commission's Trip Record Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). This dataset contains over 2 million samples of yellow cab rides, and the goal is to predict whether a trip will result in a tip greater than 20% or not.

**Dataset features**
* **`passenger_count`**
    * Float (whole number) representing number of passengers.
* **`trip_distance`** 
    * Float representing trip distance in miles.
* **`fare_amount`**
    * Float representing total price including tax, tip, fees, etc.
* **`trip_duration`**
    * Integer representing seconds elapsed.
* **`hour`**
    * Hour that the trip started.
    * Integer in the range `[0, 23]`
* **`day_of_week`**
    * Integer in the range `[1, 7]`.
* **`is_big_tip`**
    * Whether the tip amount was greater than 20%.
    * Boolean `[True, False]`.


```python
import ray

from ray.train import ScalingConfig, RunConfig
from ray.train.xgboost import XGBoostTrainer
from ray.train.xgboost import XGBoostPredictor

from ray import tune
from ray.tune import Tuner, TuneConfig

from ray import serve
from starlette.requests import Request

import requests, json
import numpy as np
```

__Read, preprocess with Ray Data__


```python
ray.init()
```


```python
dataset = ray.data.read_parquet("s3://anonymous@anyscale-training-data/intro-to-ray-air/nyc_taxi_2021.parquet").repartition(16)

train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)
```

__Fit model with Ray Train__


```python
trainer = XGBoostTrainer(
    label_column="is_big_tip",
    scaling_config=ScalingConfig(num_workers=4, use_gpu=False),
    params={ "objective": "binary:logistic", },
    datasets={"train": train_dataset, "valid": valid_dataset},
    run_config=RunConfig(storage_path='/mnt/cluster_storage/')
)

result = trainer.fit()
```

__Optimize hyperparameters with Ray Tune__


```python
tuner = Tuner(trainer, 
            param_space={'params' : {'max_depth': tune.randint(2, 12)}},
            tune_config=TuneConfig(num_samples=3, metric='train-logloss', mode='min'),
            run_config=RunConfig(storage_path='/mnt/cluster_storage/'))

checkpoint = tuner.fit().get_best_result().checkpoint
```

__Batch inference with Ray Data__


```python
class OfflinePredictor:
    def __init__(self):
        import xgboost
        self._model = xgboost.Booster()
        self._model.load_model(checkpoint.path + '/model.json')

    def __call__(self, batch):
        import xgboost
        import pandas as pd
        dmatrix = xgboost.DMatrix(pd.DataFrame(batch))    
        outputs = self._model.predict(dmatrix)
        return {"prediction": outputs}
```


```python
predicted_probabilities = valid_dataset.drop_columns(['is_big_tip']).map_batches(OfflinePredictor, compute=ray.data.ActorPoolStrategy(size=2))
```


```python
predicted_probabilities.take_batch()
```

__Online prediction with Ray Serve__


```python
@serve.deployment
class OnlinePredictor:
    def __init__(self, checkpoint):
        import xgboost
        self._model = xgboost.Booster()
        self._model.load_model(checkpoint.path + '/model.json')        
        
    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        data = json.loads(data)
        return {"prediction": self.get_response(data) }
    
    def get_response(self, data):
        import pandas as pd
        import xgboost
        dmatrix = xgboost.DMatrix(pd.DataFrame(data, index=[0])) 
        return self._model.predict(dmatrix)

handle = serve.run(OnlinePredictor.bind(checkpoint=checkpoint))
```


```python
sample_input = valid_dataset.take(1)[0]
del(sample_input['is_big_tip'])
del(sample_input['__index_level_0__'])

requests.post("http://localhost:8000/", json=json.dumps(sample_input)).json()
```


```python
serve.shutdown()
```
