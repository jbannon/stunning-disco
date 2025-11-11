##
#
#
# Lightweight imports. 
#
#
####

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


dim_range = [50,100,1000,5000]
sample_range = [50,100,500]

seed = 314159

rng = np.random.default_rng(seed=seed)

for n_samples, n_feat in product(sample_range,dim_range):
	
	# generate the data. This can be pulled from your csv. 
	data = rng.normal(0,10,size=(n_samples,n_feat))
	
	# for scikit learn to make the graph need to treat features like 'samples' in their documentation
	data = data.transpose()

	if n_samples == 500 and n_feat == 5000:
		df = pd.DataFrame(data)
		df.to_csv("simdata/fake_data.csv")

	if n_feat <1000:
		k_range = [2,5,10, 50, n_feat]
	else:
		k_range = [5]
	
	for k in k_range:
		print(f"For {n_samples} samples with {n_feat} features using {k} neighbors...")
		if k<n_feat:
			
			# we create two adjacency matrices, one unweighted and one with distance weights

			A =  kneighbors_graph(data, k, mode='connectivity', include_self=False).toarray()
			A_weighted =  kneighbors_graph(data, k, mode='distance', include_self=False).toarray()
			
			# turn the matrices into graph objects
			G = nx.from_numpy_array(A)
			G_weighted = nx.from_numpy_array(A_weighted)

			
			
			# Community detection on the unweighted graph
			s = time.time()
			comp = nx.community.louvain_communities(G)
			e = time.time()
			print(f"\tLouvain on the unweighted graph took: {(e-s)} seconds")
			

			s = time.time()
			comp = nx.community.greedy_modularity_communities(G)
			e = time.time()
			print(f"\tGreedy Modularity on the unweighted graph took: {(e-s)} seconds")


			s = time.time()
			comp = nx.community.louvain_communities(G_weighted)
			e = time.time()
			print(f"\tLouvain on the weighted graph took: {(e-s)/60} seconds")
			

			s = time.time()
			comp = nx.community.greedy_modularity_communities(G_weighted)
			e = time.time()
			print(f"\tGreedy Modularity on the weighted graph took: {(e-s)} seconds")
			
		else:
			# try with all features connected by distance
			A_weighted = pairwise_distances(data)
			G_weighted = nx.from_numpy_array(A_weighted)
			

			s = time.time()
			comp = nx.community.louvain_communities(G_weighted)
			e = time.time()
			print(f"\tLouvain on the unweighted graph took: {(e-s)} seconds")
			

			s = time.time()
			comp = nx.community.greedy_modularity_communities(G)
			e = time.time()
			print(f"\tGreedy Modularity on the unweighted graph took: {(e-s)} seconds")


		print("\n")
			
	
	

