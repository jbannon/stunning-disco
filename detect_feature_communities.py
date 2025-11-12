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


# read in our data which we assume has features as columns, samples as rows.
data = pd.read_csv("simdata/fake_data.csv",index_col=0)


####
#
#
#	IMPORTANT HYPERPARAMS TO SET
#
#
#######
num_nbrs = 50
weighted = True
corr_weights = True
corr_thresh = 0.1 # needed to sparsify
print_members = False
seed = 69420

# change behavior to reflect chosen parameters 
mode = 'distance' if weighted else 'connectivity'
weight_field = None if not weighted else "weight"


# We use sklearns graph construction tools to build the graph
# and they expect to cluster the rows. Thus we compute the transpose
X = data.values.transpose() 

p = np.corrcoef(X)
# print(p.shape)
# sys.exit()
## now we do the processing


# make the adjacency matrix

if corr_weights:
	# need to re-flip X
	# shift b/c we expect positive weights
	A = np.corrcoef(X.transpose())

	# if the (absolute) correlation is below your threshold remove it
	# this sparsifies the graph and will improve performance
	# and also prevent potentially spurious communities

	kill_idxs = np.where(np.abs(A)<=corr_thresh)
	A[kill_idxs] = 0
	np.fill_diagonal(A,0)

else:

	A =  kneighbors_graph(X, num_nbrs, mode=mode, include_self=False).toarray()

	
# now we turn that adjaceny matrix into a networkx graph object
G = nx.from_numpy_array(A)


# run the community detection on the potentially weighted graph
comms = nx.community.louvain_communities(G, weight= weight_field,seed=seed )


print(f"Number of communities: {len(comms)}")
for i in range(len(comms)):
	print(f"Community {i} has {len(comms[i])} members. ")
	if print_members:
		print("The members are....")
		print(comms[i])
		# community members will now correspond to columns of the data aka
		# the features you want to cluster. 
	print("\n")

	