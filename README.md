# PCM

## Setup
Update your PYTHONPATH environment variable to include the root of the project

`export PYTHONPATH=${PYTHONPATH}:$ROOT`
where `$ROOT` is the full path to the root of this repo.

## Usage
### multi_robot_optimization
`python multi_robot_optimization.py input.g2o output.g2o`
### build_adjacency
`python build_adjacency.py datasets/city10000.g2o`

where `input.g2o` is a multi-robot g2o file, and `output.g2o` will be the optimized one.

## Style Guidelines
* python: <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>
* C++: <https://google.github.io/styleguide/cppguide.html>
