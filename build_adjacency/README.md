# How to use

Requires: numpy, scipy

	python3 build_adjacency.py ../datasets/manhattanOlson3500.g2o mah_adjacency.mtx

Go to ../fmc/src/

run ./fmc ../../build_adjacency/adjacency.mtx -t 1

Note: added a flag to AdjacencyMatrix class to backward support the initial building process (without optimization). When setting `optim=True`, embedding optimization from gtsam_optimize.