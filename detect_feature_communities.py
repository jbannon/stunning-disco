import networkx as nx
# *** if performance is an issue look into the networkit package***

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import time
import sys

data = pd.read_csv("simdata/fake_data.csv",index_col=0)
X = data.values.transpose()
# sys.exit()
num_nbrs = 50
weighted = True

mode = 'distance' if weighted else 'connectivity'



A =  kneighbors_graph(X, num_nbrs, mode=mode, include_self=False).toarray()
	

G = nx.from_numpy_array(A)
weight_field = None if not weighted else "weight"
comms = nx.community.louvain_communities(G, weight= weight_field )


print(f"Number of communities: {len(comms)}")
for i in range(len(comms)):
	print(f"Community {i} has {len(comms[i])} members and they are...")
	print(comms[i])
	print("\n")

	# community members will now correspodn to columns of the data aka
	# the features you want to cluster. 