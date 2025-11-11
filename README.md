## Clustering Radiomics Features



This cute lil' repo exists as an example of a way to use community detection on graphs to perform unsupervised clustering of features. 

##### Community Detection on a Bumper Sticker


A *graph* is a collection of objects that are related in pairs. In mathy terms it's a set of *nodes* connected by *edges.* The problem of *community detection* is, as the name suggests, about finding communities in graphs. 


As an example, consider a graph where the nodes are members of the BHKLab and edges exist between people if they've exchanged messages on Slack (these edges could be weighted to reflect frequency of communication). A community detection task would be to see if you could guess from this data who is in radiomics and who is in pharmacogenomics. 



Generally speaking, exact community detection is **impossible** (NP-Hard). There are, however, many many heuristic algorithms that can work. This repository looks at three such algorithms:

	


##### Detecting Clusters of Features

This repo contains a pixi environment (written on Mac, sorry) and two scripts `time_comm_det.py` which simulates data and does some community detection using two methods, Louvain and Greedy Modularity, while timing them. Greedy Modularity is MUCH slower at the scale we care about. 

The other script `detect_feature_communities.py` is a very simple script to use Louvain clustering to cluster the features. 