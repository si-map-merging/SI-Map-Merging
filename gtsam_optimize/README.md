# Optimize graph with GTSAM

## Requirements
* Install custom GTSAM
* Install GTSAM python per instruction in its `cython` folder
------------------------------------------------
## Python Usage


------------------------------------------------
## C++ Usage
## Build
1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`

## Usage
1. In `build` directory:

   * `./OptimizeG2o input.g2o output.g2o`

     This will optimize the input graph, and output it as `output.g2o`.
