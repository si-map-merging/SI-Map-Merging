# Scale Invariant Multi-agent Map Merging

Method for merging graphs from multiple agents where the scales between them are unknown.

## Dependencies
* Modified version of GTSAM with python bindings (hosted at https://github.mit.edu/vnav-project/gtsam)
* numpy
* numpy-quaternion
* sophuspy
* scipy
* numba
* tqdm
* fmc in `find_max_clique/fmc`

## Setup
Update your PYTHONPATH environment variable to include the root of the project

`export PYTHONPATH=${PYTHONPATH}:$ROOT`
where `$ROOT` is the full path to the root of this repo.

------------------------------------

## Usages
### Create a multi-agent g2o file from a single-agent g2o file
1. `cd process_g2o`
2. `python3 convert_to_multi.py ../datasets/manhattanOlson3500.g2o output.g2o`

This will create a multi-agent g2o called `output.g2o` from the `manhattanOlson3500.g2o` dataset.

-----------------------------------

### Perform pose graph optimization on a multi-agent graph
`python3 multi_agent_optimization.py process_g2o/output.g2o output.g2o`

This will perform the pose graph optimization on the multi-agent graph in `process_g2o/output.g2o`, and output as `output.g2o` in current directory.

-----------------------------------

### Build Adjacency
`python3 build_adjacency.py datasets/city10000.g2o`

where `input.g2o` is a multi-agent g2o file, and `output.g2o` will be the optimized one.

--------------------------------------------------

## Style Guidelines
* python: <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>
* C++: <https://google.github.io/styleguide/cppguide.html>
