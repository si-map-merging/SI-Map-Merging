"""Run fmc executable and print max clique nodes
"""

import os

def find_max_clique(fmc_path, input_fpath, heuristic=True):
    """Call fmc and get max clique nodes

    Args:
        fmc_path: fmc executable file path
        input_fpath: input mtx file path
        heuristic: whether use the heursitic version of fmc
    Returns:
        nodes: sorted indices of the max clique nodes
    """
    command = fmc_path + " " + input_fpath + " -p"
    algorithm_type = 0
    if heuristic:
        algorithm_type = 1
    command += " -t {}".format(algorithm_type)
    output = os.popen(command).read()

    nodes = []
    for line in output.splitlines():
        token = "Maximum clique:"
        if line.startswith(token):
            nodes = [int(x) for x in line[len(token):].split()]
    nodes.sort()
    return nodes

if __name__ == "__main__":
    nodes = find_max_clique("fmc/src/fmc", "fmc/testgraphs/hamming6-2.clq.mtx")
    print(nodes)