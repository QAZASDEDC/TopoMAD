# TopoMAD

## Datasets

### MBD

 [MBD (1).csv](https://github.com/QAZASDEDC/TopoMAD/blob/master/DatasetUpdate/MBD%20(1).csv) : metrics of MBD
 
 [MBD_topology.pk](https://github.com/QAZASDEDC/TopoMAD/blob/master/DatasetUpdate/MBD_topology.pk): topology of MBD reprensented as an edge array

### MMS

 [MMS (1).csv](https://github.com/QAZASDEDC/TopoMAD/blob/master/DatasetUpdate/MMS%20(1).csv) : metrics of MMS

 [MMS_topology.pk](https://github.com/QAZASDEDC/TopoMAD/blob/master/DatasetUpdate/MMS_topology.pk) : topology of MMS reprensented as an edge array
 
## How to Open the Datasets

## MBD
```
DATASET = "../DatasetUpdate/MBD (1).csv"
TOPOLOGY = "../DatasetUpdate/MBD_topology.pk"

import pandas as pd
import numpy as np
import pickle

data = pd.read_csv(DATASET, header=[0,1])

# preprocess
metric = data.drop(['date', 'label'], axis = 1)
metric.columns.names = ['host','metric']
tempm = metric.swaplevel('metric','host',axis=1).stack()
tempm = (tempm-tempm.mean())/(tempm.std())
metric = tempm.unstack().swaplevel('metric','host',axis=1).stack().unstack()

with open(TOPOLOGY, 'rb') as f:
    edge_index = pickle.load(f)
```
edge_index represents graph connectivity in COO format with shape \[2, num_edges\]. In edge_index, each node in the topology is represented with its corresponding index. Specifically, the index $i$ corresponds to the node named as the $i^{th}$ object in 
```metric.columns.levels[0]```

## MMS
```
DATASET = "../DatasetUpdate/MMS (1).csv"
TOPOLOGY = "../DatasetUpdate/MMS_topology.pk"

import pandas as pd
import numpy as np
import pickle

data = pd.read_csv(DATASET, header=[0,1])

# preprocess
metric = data.drop(['TimeStamp', 'label'], axis = 1)
metric.columns.names = ['pod','metric']
tempm = metric.swaplevel('metric','pod',axis=1).stack()
tempm = (tempm-tempm.mean())/(tempm.std())
metric = tempm.unstack().swaplevel('metric','pod',axis=1).stack().unstack()

with open(TOPOLOGY, 'rb') as f:
    edge_index = pickle.load(f)
```
edge_index represents graph connectivity in COO format with shape \[2, num_edges\]. In edge_index, each node in the topology is represented with its corresponding index. Specifically, the index $i$ corresponds to the node named as the $i^{th}$ object in 
```metric.columns.levels[0]```
