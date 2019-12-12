"""Convert a single-robot g2o file to a multi-robot g2o file

Example usages:
  $ python3 convert_to_multi.py input.g2o output.g2o
"""
import argparse
from utils import SingleRobotGraph2D, SingleRobotGraph3D
import numpy as np


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Convert a single-robot g2o file to a multi-robot g2o file")
    parser.add_argument("input_fpath", metavar="input.g2o", type=str,
                        help="input g2o file path")
    parser.add_argument("output_fpath", metavar="output.g2o", type=str,
                        nargs='?', default="output.g2o",
                        help="output g2o file path")
    parser.add_argument("--3D", dest="is_3D", action="store_true", help="whether input is 3D")
    parser.add_argument("--max_inter_lc",default=6,type=int,help="max_inter_lc")
    parser.add_argument("--random_inter_lc",default=18,type=int,help="random_inter_lc")
    args = parser.parse_args()

    # Construct graph from g2o file
    if args.is_3D:
      graph = SingleRobotGraph3D()
    else:
      graph = SingleRobotGraph2D()
    graph.read_from(args.input_fpath)

    print("========== Input g2o Graph Summary ================")
    graph.print_summary()

    multi_graph = graph.to_multi(n_max_inter_lc=args.max_inter_lc)
    print("========== Multi Robot g2o Graph Summary ================")
    multi_graph.print_summary()

    if not args.is_3D:
      aliasing_lc = [2,3]
      multi_graph.add_perceptual_aliasing_lc(aliasing_lc[0], aliasing_lc[1])
      sb = np.random.uniform(0.5, 2)
      multi_graph.vary_scale(sb)
      print('real_sb ',sb, 'record')
    multi_graph.add_random_inter_lc(N=args.random_inter_lc)
    num_inliers = args.max_inter_lc
    num_outliers = args.random_inter_lc+aliasing_lc[0]*aliasing_lc[1]
    print('outlier_rate', num_outliers/float(num_inliers+num_outliers), 'record')
    print("========== Noisy Multi Robot g2o Graph Summary ================")
    multi_graph.print_summary()
    multi_graph.write_to(args.output_fpath)
