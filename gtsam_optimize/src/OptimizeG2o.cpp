/**
 * @file OptimizeG2o.cpp
 *
 * @brief Read a g2o file, optimize it, and output as another g2o file
 *        TODO(Jay) Currently assumes 2D, extend to 3D
 *
 * @author Jay Li
 */

#include <iostream>

#include <gtsam/slam/dataset.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

using namespace std;
using namespace gtsam;

int main(const int argc, const char *argv[]) {

  // Parse input g2o path
  assert (argc > 1);
  string inputFPath = argv[1];

  // Parse output g2o path
  string outputFPath = "output.g2o";
  if (argc > 2) {
    outputFPath = argv[2];
  }

  // TODO(Jay) parse kernel
  string kernelType = "none";

  // TODO(Jay) parse maxIteration
  int maxIterations = 100;

  // Read file and create factor graph
  NonlinearFactorGraph::shared_ptr graph;
  Values::shared_ptr initial;
  bool is3D = false;
  // TODO(Jay) Know the difference between different kernels and add them
  assert(kernelType == "none");
  if (kernelType.compare("none") == 0) {
    boost::tie(graph, initial) = readG2o(inputFPath, is3D);
  }

  // Add prior on pose 0
  NonlinearFactorGraph graphWithPrior = *graph;
  noiseModel::Diagonal::shared_ptr priorModel = noiseModel::Diagonal::Variances(
    Vector3(1e-6, 1e-6, 1e-8));
  graphWithPrior.add(PriorFactor<Pose2>(0, Pose2(), priorModel));

  // Optimize the graph
  cout << "Start optimization\n";
  GaussNewtonParams params;
  params.maxIterations = maxIterations;
  params.setVerbosity("TERMINATION");
  GaussNewtonOptimizer optimizer(graphWithPrior, *initial, params);
  Values result = optimizer.optimize();

  // Print optimization stats
  cout << "Optimization completed\n";
  cout << "Initial error: " << graph->error(*initial) << endl;
  cout << "Final error: " << graph->error(result) << endl;

  // Write to file
  cout << "Writing results to file: " << outputFPath << endl;
  NonlinearFactorGraph::shared_ptr graphNoKernel;
  Values::shared_ptr initial2;
  boost::tie(graphNoKernel, initial2) = readG2o(inputFPath);
  writeG2o(*graphNoKernel, result, outputFPath);
  cout << "Done!" << endl;

  return 0;
}