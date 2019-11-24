# PCM

## Steps
1. Generate adjacency matrix by `generate_adjacency.py input.g2o adj.mtx`.
2. Find max clique by `./find_max_clique adj.mtx fmc.mtx`.
3. Optimize multi robot graph with trusted inter-robot loop closures `optimize_trusted_multi.py input.g2o fmc.mtx output.g2o`.

These can be done in a single line by `./pcm.sh input.g2o output.g2o`.

## Style Guidelines
* python: <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>
* C++: <https://google.github.io/styleguide/cppguide.html>