"""Given a multi robot g2o file, obtain trusted inter-robot loop closures, and
optimize the overall graph

Example usages:
    TODO
"""

import argparse
from scipy import io
from process_g2o.utils import MultiRobotGraph
from find_max_clique.find_max_clique import find_max_clique
from gtsam_optimize import optimization
from build_adjacency.build_adjacency import AdjacencyMatrix


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

    # # Separate into two single robot graphs
    # single_graphs = multi_graph.to_singles()
    # for i, graph in enumerate(single_graphs):
    #     print("========== Single Robot {} Graph Summary ===========".format(i))
    #     graph.print_summary()

    # # Feed graphs to GTSAM
    # for robot_i, graph in enumerate(single_graphs):
    #     gtsam_graph = optimization.Graph(graph)
    #     gtsam_graph.optimize()
    #     print("===== Single Robot {} Graph Optimization =====".format(robot_i))
    #     gtsam_graph.print_stats()

    # Compute Jacobian => Covariances

    # Compute consistency matrix
    adj = AdjacencyMatrix(multi_graph, gamma=0.1)

    # Compute Adjacency matrix
    coo_adj_mat = adj.build_adjacency_matrix()
    mtx_fpath = "adj.mtx"
    io.mmwrite(mtx_fpath, coo_adj_mat, symmetry='symmetric')

    # print("inital lc:\n")
    # for i, edge in enumerate(adj.inter_lc_edges, 1):
    #     print("{}: {}".format(i, edge))

    # Call fmc on the adjacency matrix, to get trusted inter-robot loop closures
    fmc_path = "find_max_clique/fmc/src/fmc"
    trusted_lc_indices = find_max_clique(fmc_path, mtx_fpath)
    print("# trusted lc: {}".format(len(trusted_lc_indices)))
    print(trusted_lc_indices)
    trusted_lc = adj.get_trusted_lc(trusted_lc_indices)
    multi_graph.set_inter_lc(trusted_lc)
    multi_graph.write_to(args.output_fpath)

    # # Perform overall graph optimization
    # merged_graph = multi_graph.merge_to_single()
    # gtsam_graph = optimization.Graph(merged_graph)
    # gtsam_graph.optimize()
    # print("===== Multi-Robot Optimization =====")
    # gtsam_graph.print_stats()

    # # Write result as g2o
    # gtsam_graph.write_to(args.output_fpath)