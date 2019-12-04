"""Convert a single-robot g2o file to a multi-robot g2o file

Example usages:
  $ python3 convert_to_multi.py input.g2o output.g2o
"""

import argparse
from utils import SingleRobotGraph

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert a single-robot g2o file to a multi-robot g2o file")
    parser.add_argument("input_fpath", metavar="input.g2o", type=str,
                        help="input g2o file path")
    parser.add_argument("output_fpath", metavar="output.g2o", type=str,
                        nargs='?', default="output.g2o",
                        help="output g2o file path")
    args = parser.parse_args()

    # Construct graph from g2o file
    graph = SingleRobotGraph()
    graph.read_from(args.input_fpath)

    print("========== Input g2o Graph Summary ================")
    graph.print_summary()

    multi_graph = graph.to_multi(n_max_inter_lc=20)
    print("========== Multi Robot g2o Graph Summary ================")
    multi_graph.print_summary()

    multi_graph.add_random_inter_lc(N=20)
    print("========== Noisy Multi Robot g2o Graph Summary ================")
    multi_graph.print_summary()
    multi_graph.write_to(args.output_fpath)
