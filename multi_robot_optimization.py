"""Given a multi robot g2o file, obtain trusted inter-robot loop closures, and
optimize the overall graph

Example usages:
    TODO
"""

import argparse
from process_g2o.utils import MultiRobotGraph
from find_max_clique.find_max_clique import find_max_clique
from gtsam_optimize import optimization


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate the adjacency matrix from a multi robot g2o")
    parser.add_argument("input_fpath", metavar="input.g2o", type=str,
                        help="input g2o file path")
    parser.add_argument("output_fpath", metavar="output.g2o", type=str,
                        nargs='?', default="output.g2o",
                        help="output g2o file path")
    args = parser.parse_args()

    # Construct multi robot graph from g2o file
    multi_graph = MultiRobotGraph()
    multi_graph.read_from(args.input_fpath)

    print("========== Multi Robot Graph Summary ==============")
    multi_graph.print_summary()

    # Separate into two single robot graphs
    single_graphs = multi_graph.to_single()
    for i, graph in enumerate(single_graphs):
        print("========== Single Robot {} Graph Summary ===========".format(i))
        graph.print_summary()

    # Feed graphs to GTSAM
    for robot_i, graph in enumerate(single_graphs):
        gtsam_graph = optimization.Graph(graph)
        gtsam_graph.optimize()
        print("===== Single Robot {} Graph Optimization =====".format(robot_i))
        gtsam_graph.print_stats()

    # Compute Jacobian => Covariances

    # Compute consistency matrix

    # Compute Adjacency matrix
    # mtx_fpath = "adj.mtx"

    # Call fmc on the adjacency matrix, to get trusted inter-robot loop closures
    fmc_path = "find_max_clique/fmc/src/fmc"
    # trusted_lc = find_max_clique(fmc_path, mtx_fpath)

    # Perform overall graph optimization