# Datasets

## To view a dataset
1. Build `g2o` from this link: <https://github.com/RainerKuemmerle/g2o>
2. In its directory `bin`,
   - run `./g2o_viewer [g2o file path]`
   - (OR) run `./g2o_viewer`, and use its GUI to load a g2o file.

## 2D datasets
- `manhattanOlson3500.g2o`

  Creator: Ed Olson

  See "Fast Iterative Alignment of Pose Graphs with Poor Initial Estimates",
  Edwin Olson, John Leonard and Seth Teller, ICRA 2006

  Simulated pose graph.

  3500 poses and 5598 constraints (3499 odometry and 2099 loop closings)

- `city10000.g2o`

  Creator: Hordur Johannsson and Michael Kaess

  Simulated pose graph.

  10000 poses, 20687 constraints (9999 odometry and 10688 loop closings)

## Ground truth
- Some of the datasets have ground truth, stored in `groundtruth` directory, obtained from: <https://github.com/rickytan/iSAM/tree/master/data/groundtruth>