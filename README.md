# Scale-invariant PCM

Pairwise Consistent Measurement Set Maximization for Robust Multi-robot Map Merging, where the Scales between Robots are unknown.

## Dependencies
* GTSAM with python bindings
* numpy
* numpy-quaternion
* sophuspy
* scipy

## Setup
Update your PYTHONPATH environment variable to include the root of the project

`export PYTHONPATH=${PYTHONPATH}:$ROOT`
where `$ROOT` is the full path to the root of this repo.

------------------------------------

## Usages
### Create a multi-robot g2o file from a single-robot g2o file
1. `cd process_g2o`
2. `python3 convert_to_multi.py ../datasets/manhattanOlson3500.g2o output.g2o`

This will create a multi-robot g2o called `output.g2o` from the `manhattanOlson3500.g2o` dataset.

-----------------------------------

### Perform pose graph optimization on a multi-robot graph
`python3 multi_robot_optimization.py process_g2o/output.g2o output.g2o`

This will perform the pose graph optimization on the multi-robot graph in `process_g2o/output.g2o`, and output as `output.g2o` in current directory.

-----------------------------------

### Build Adjacency
`python3 build_adjacency.py datasets/city10000.g2o`

where `input.g2o` is a multi-robot g2o file, and `output.g2o` will be the optimized one.

--------------------------------------------------

## Style Guidelines
* python: <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>
* C++: <https://google.github.io/styleguide/cppguide.html>
