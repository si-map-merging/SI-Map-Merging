"""Given a multi robot g2o file, obtain trusted inter-robot loop closures, and
optimize the overall graph

Example usages:
    python3 multi_agent_optimization input.g2o output.g2o
"""

import argparse
from scipy import io
from process_g2o.utils import MultiRobotGraph2D, MultiRobotGraph3D
from find_max_clique.find_max_clique import find_max_clique
from gtsam_optimize import optimization
from build_adjacency.build_adjacency import AdjacencyMatrix, AdjacencyMatrix3D


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate the adjacency matrix from a multi robot g2o")
    parser.add_argument("input_fpath", metavar="input.g2o", type=str,
                        help="input g2o file path")
    parser.add_argument("output_fpath", metavar="output.g2o", type=str,
                        nargs='?', default="output.g2o",
                        help="output g2o file path")
    parser.add_argument("--3D", dest="is_3D", action="store_true",
                        help="whether input is 3D")
    args = parser.parse_args()

    # Construct multi robot graph from g2o file
    if args.is_3D:
        multi_graph = MultiRobotGraph3D()
    else:
        multi_graph = MultiRobotGraph2D()
    multi_graph.read_from(args.input_fpath)

    print("========== Multi Robot Graph Summary ==============")
    multi_graph.print_summary()

    # Compute consistency matrix
    if args.is_3D:
        adj = AdjacencyMatrix3D(multi_graph, gamma=0.1, optim=True)
    else:
        adj = AdjacencyMatrix(multi_graph, gamma=1, optim=True)
    # Compute Adjacency matrix
    coo_adj_mat = adj.build_adjacency_matrix()
    mtx_fpath = "adj.mtx"
    io.mmwrite(mtx_fpath, coo_adj_mat, field='integer', symmetry='symmetric')

    print("initial lc:\n")
    for i, edge in enumerate(adj.inter_lc_edges, 1):
        print("{}: {}".format(i, edge))

    total_lc = adj.inter_lc_n
    positives = 0
    for edge in adj.inter_lc_edges:
        if not edge.is_outlier:
            positives += 1
    negatives = total_lc - positives


    # Call fmc on the adjacency matrix, to get trusted inter-robot loop closures
    fmc_path = "find_max_clique/fmc/src/fmc"
    trusted_lc_indices = find_max_clique(fmc_path, mtx_fpath)
    print("# trusted lc: {}".format(len(trusted_lc_indices)))
    print(trusted_lc_indices)
    trusted_lc = adj.get_trusted_lc(trusted_lc_indices)
    multi_graph.set_inter_lc(trusted_lc)
    multi_graph.write_to(args.output_fpath)

    true_pos = 0
    for edge in trusted_lc:
        if not edge.is_outlier:
            true_pos += 1
    false_pos = len(trusted_lc) - true_pos

    if positives == 0:
        TPR = 1
    else:
        TPR = true_pos / positives
    print("TPR: {}".format(TPR),'record')
    if negatives == 0:
        FPR = 1
    else:
        FPR = false_pos / negatives
    TNR = 1 - FPR
    print("TNR: {}".format(TNR),'record')

    # Perform overall graph optimization
    merged_graph = multi_graph.merge_to_single()
    gtsam_graph = optimization.Graph2D(merged_graph)
    gtsam_graph.optimize()
    print("===== Multi-Robot Optimization =====")
    gtsam_graph.print_stats()

    # Write result as g2o
    gtsam_graph.write_to(args.output_fpath)
