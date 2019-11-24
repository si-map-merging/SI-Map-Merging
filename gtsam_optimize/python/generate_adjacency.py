"""Given a multi robot g2o file, generate the adjacency matrix

Example usages:
    TODO
"""

import argparse
import sys
# A hack to include package from another directory
sys.path.insert(1, "../../process_g2o")
from utils import MultiRobotGraph

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
    graph = MultiRobotGraph()
    # graph.read_from(args.input_fpath)

    # Separate into two single robot graphs

    # Feed graphs to GTSAM

    # Optimize graphs with GTSAM

    # Compute Jacobian => Covariances

    # Compute consistency matrix

    # Compute Adjacency matrix
