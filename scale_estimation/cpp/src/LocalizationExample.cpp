/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file LocalizationExample.cpp
 * @brief Simple robot localization example, with three "GPS-like" measurements
 * @author Frank Dellaert
 */

/**
 * A simple 2D pose slam example with "GPS" measurements
 *  - The robot moves forward 2 meter each iteration
 *  - The robot initially faces along the X axis (horizontal, to the right in 2D)
 *  - We have full odometry between pose
 *  - We have "GPS-like" measurements implemented with a custom factor
 */

// We will use Pose2 variables (x, y, theta) to represent the robot positions
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>

// We will use simple integer Keys to refer to the robot poses.
#include <gtsam/inference/Key.h>

// As in OdometryExample.cpp, we use a BetweenFactor to model odometry measurements.
#include <gtsam/slam/BetweenFactor.h>

// We add all facors to a Nonlinear Factor Graph, as our factors are nonlinear.
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

// The nonlinear solvers within GTSAM are iterative solvers, meaning they linearize the
// nonlinear functions around an initial linearization point, then solve the linear system
// to update the linearization point. This happens repeatedly until the solver converges
// to a consistent set of variable values. This requires us to specify an initial guess
// for each variable, held in a Values container.
#include <gtsam/nonlinear/Values.h>

// Finally, once all of the factors have been added to our factor graph, we will want to
// solve/optimize to graph to find the best (Maximum A Posteriori) set of variable values.
// GTSAM includes several nonlinear optimizers to perform this step. Here we will use the
// standard Levenberg-Marquardt solver
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

// Once the optimized values have been calculated, we can also calculate the marginal covariance
// of desired variables
#include <gtsam/nonlinear/Marginals.h>

#include <gtsam/base/Testable.h>
#include <gtsam/base/Lie.h>

using namespace std;
using namespace gtsam;

// Before we begin the example, we must create a custom unary factor to implement a
// "GPS-like" functionality. Because standard GPS measurements provide information
// only on the position, and not on the orientation, we cannot use a simple prior to
// properly model this measurement.
//
// The factor will be a unary factor, affect only a single system variable. It will
// also use a standard Gaussian noise model. Hence, we will derive our new factor from
// the NoiseModelFactor1.
#include <gtsam/nonlinear/NonlinearFactor.h>

class UnaryFactor: public NoiseModelFactor1<Pose2> {

  // The factor will hold a measurement consisting of an (X,Y) location
  // We could this with a Point2 but here we just use two doubles
  double mx_, my_;

public:
  /// shorthand for a smart pointer to a factor
  typedef boost::shared_ptr<UnaryFactor> shared_ptr;

  // The constructor requires the variable key, the (X, Y) measurement value, and the noise model
  UnaryFactor(Key j, double x, double y, const SharedNoiseModel& model):
    NoiseModelFactor1<Pose2>(model, j),mx_(x), my_(y) {}

  virtual ~UnaryFactor() {}

  // Using the NoiseModelFactor1 base class there are two functions that must be overridden.
  // The first is the 'evaluateError' function. This function implements the desired measurement
  // function, returning a vector of errors when evaluated at the provided variable value. It
  // must also calculate the Jacobians for this measurement function, if requested.
  Vector evaluateError(const Pose2& q, boost::optional<Matrix&> H = boost::none) const
  {
    // The measurement function for a GPS-like measurement is simple:
    // error_x = pose.x - measurement.x
    // error_y = pose.y - measurement.y
    // Consequently, the Jacobians are:
    // [ derror_x/dx  derror_x/dy  derror_x/dtheta ] = [1 0 0]
    // [ derror_y/dx  derror_y/dy  derror_y/dtheta ] = [0 1 0]
    if (H) 
      (*H) = (Matrix(2,3) << 1.0,0.0,0.0, 0.0,1.0,0.0).finished();
    return (Vector(2) << q.x() - mx_, q.y() - my_).finished();
  }

  // The second is a 'clone' function that allows the factor to be copied. Under most
  // circumstances, the following code that employs the default copy constructor should
  // work fine.
  virtual gtsam::NonlinearFactor::shared_ptr clone() const {

    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
       gtsam::NonlinearFactor::shared_ptr(new UnaryFactor(*this))); 

  }
 
  // Additionally, we encourage you the use of unit testing your custom factors,
  // (as all GTSAM factors are), in which you would need an equals and print, to satisfy the
  // GTSAM_CONCEPT_TESTABLE_INST(T) defined in Testable.h, but these are not needed below.

}; // UnaryFactor



template<class VALUE>
class SIBetweenFactor: public NoiseModelFactor2<VALUE, VALUE> {

  Matrix H_;
  Matrix eye_T;
public:

  typedef VALUE T;

private:

  typedef SIBetweenFactor<VALUE> This;
  typedef NoiseModelFactor2<VALUE, VALUE> Base;

  VALUE measured_; /** The measurement */
  
  /** concept check by type */
  GTSAM_CONCEPT_LIE_TYPE(T)
  GTSAM_CONCEPT_TESTABLE_TYPE(T)

public:

  // shorthand for a smart pointer to a factor
  typedef typename boost::shared_ptr<SIBetweenFactor> shared_ptr;

  /** default constructor - only use for serialization */
  SIBetweenFactor() {}

  /** Constructor */
  SIBetweenFactor(Key key1, Key key2, const VALUE& measured,
      const SharedGaussian& model) :
    Base(model, key1, key2), measured_(measured) {
      auto size = measured_.dimension;
      // H_ = eye(size);

      auto pose_trans = measured_.translation();//.vector();
      auto zero_translation = measured_.translation().identity();
      auto zero_rotation = measured_.rotation().identity();
      auto zero_pose = VALUE(zero_rotation, zero_translation);
      auto x = zero_pose.localCoordinates(VALUE(zero_rotation, pose_trans));
      auto dim  = measured_.translation().dimension;
      

      

      // translation()
      // gtsam::noiseModel::Gaussian::shared_ptr gaussian_model(model);
      double denominator = model->Mahalanobis(x);
      Matrix n1  = (x * model->whiten(x).transpose());
      Matrix n2 = (model->Whiten(eye(size)));
      // auto nominator = (x * model->whiten(x).transpose()) * model->Whiten(eye(size));
      auto nominator = n1*n2;

      auto H_trans = nominator / denominator;
      H_ = eye(size) - H_trans;
      eye_T = eye(size);
      //H_.block< >;

      cout << "x: " << x << endl;
      cout << "no: " << (x * (model->whiten(x)).transpose()) <<endl<<endl << model->Whiten(eye(size))<<endl<<endl<<nominator<<endl;
      // cout << "x*x^T: " << x*(x.transpose()) << endl;
      cout << "H_: " << H_ << endl;
  }

  virtual ~SIBetweenFactor() {}

  /// @return a deep copy of this factor
  virtual gtsam::NonlinearFactor::shared_ptr clone() const {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this))); }

  /** implement functions needed for Testable */

  /** print */
  virtual void print(const std::string& s, const KeyFormatter& keyFormatter = DefaultKeyFormatter) const {
    std::cout << s << "SIBetweenFactor("
        << keyFormatter(this->key1()) << ","
        << keyFormatter(this->key2()) << ")\n";
    measured_.print("  measured: ");
    this->noiseModel_->print("  noise model: ");
  }

  /** equals */
  virtual bool equals(const NonlinearFactor& expected, double tol=1e-9) const {
    const This *e =  dynamic_cast<const This*> (&expected);
    return e != NULL && Base::equals(*e, tol) && this->measured_.equals(e->measured_, tol);
  }

  /** implement functions needed to derive from Factor */

  /** vector of errors */
  Vector evaluateError(const T& p1, const T& p2,
      boost::optional<Matrix&> H1 = boost::none, boost::optional<Matrix&> H2 =
          boost::none) const {
    //T hx = p1.between(p2, H1, H2); // h(x)
    T result = p1.inverse() * p2;
    if (H1)
      *H1 = H_ * (-result.inverse().AdjointMap());
    if (H2)
      *H2 = H_ * eye_T;
    // cout<<"o: "<<*H1<<endl;
    // Matrix newH1 = H_ * (*H1);
    // Matrix newH2 = H_ * (*H2);
    // cout<<"n: "<<*H1<<endl;
    // *H1 = newH1;
    // *H2 = newH2;
    //hx = H_ * hx;
    // manifold equivalent of h(x)-z -> log(z,h(x))
    return H_ * measured_.localCoordinates(result);
  }

  /** return the measured */
  const VALUE& measured() const {
    return measured_;
  }

  /** number of variables attached to this factor */
  std::size_t size() const {
    return 2;
  }

private:

  /** Serialization function */
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int version) {
    ar & boost::serialization::make_nvp("NoiseModelFactor2",
        boost::serialization::base_object<Base>(*this));
    ar & BOOST_SERIALIZATION_NVP(measured_);
  }
}; // \class SIBetweenFactor



int main(int argc, char** argv) {

  // 1. Create a factor graph container and add factors to it
  NonlinearFactorGraph graph;

  // 2a. Add odometry factors
  // For simplicity, we will use the same noise model for each odometry factor
  static Matrix R = ((Matrix(3, 3) <<
    0.2*0.2, 0.0, 0.0,
    0.0, 0.2*0.2, 0.0,
    0.0, 0.0, 0.1*0.1)).finished();
  // noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));

  noiseModel::Gaussian::shared_ptr odometryNoise = noiseModel::Gaussian::Covariance(R);
  // Create odometry (Between) factors between consecutive poses
  graph.emplace_shared<SIBetweenFactor<Pose2> >(1, 2, Pose2(1.0, 0.0, 0.0), odometryNoise);
  graph.emplace_shared<SIBetweenFactor<Pose2> >(2, 3, Pose2(-1.0, 1.0, 0.0), odometryNoise);
  graph.emplace_shared<SIBetweenFactor<Pose2> >(3, 1, Pose2(0.0, -1.0, 0.0), odometryNoise);

  // static Matrix cao = ((Matrix(3, 3) <<
  //   0.2*0.2, 0.0, 0.0,
  //   0.0, 0.2*0.2, 0.0,
  //   0.0, 0.0, 0.)).finished();
  
  // noiseModel::Gaussian::shared_ptr sbnoise = noiseModel::Gaussian::SqrtInformation(cao);
  // auto sb = Pose2(2.0, 0.0, 0.0);
  // Pose3 sb2(Rot3(), Point3(0, 0, 0));
  Pose3 sb2(Rot3::RzRyRx(1., 2., 3.), Point3(0, 0, 0));
  //auto sb2 = Pose3(2.0, 0.0, 0.1,1,2,3);
  auto r = sb2.rotation().identity();
  cout << "rot: " << r.matrix() <<endl << "raw: " << sb2.rotation().matrix()  << endl;
  // auto ss = sb.localCoordinates(sb2);
  // auto jb = sbnoise->Mahalanobis(ss);
  // // auto jb = sb.vector();
  // cout <<  sbnoise->R() <<endl;
  // cout << "pose: " << ss <<endl << "mah: " << jb  << endl;

  // 2b. Add "GPS-like" measurements
  // We will use our custom UnaryFactor for this.
  noiseModel::Diagonal::shared_ptr unaryNoise = noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1)); // 10cm std on x,y
  graph.emplace_shared<UnaryFactor>(1, 0.0, 0.0, unaryNoise);
  graph.emplace_shared<UnaryFactor>(2, 2.0, 0.0, unaryNoise);
  // graph.emplace_shared<UnaryFactor>(3, 0.0, 2.0, unaryNoise);
  // graph.print("\nFactor Graph:\n"); // print

  // 3. Create the data structure to hold the initialEstimate estimate to the solution
  // For illustrative purposes, these have been deliberately set to incorrect values
  Values initialEstimate;
  initialEstimate.insert(1, Pose2(15.0, -29.0, 0.2));
  initialEstimate.insert(2, Pose2(-23.3, 12.1, -0.2));
  initialEstimate.insert(3, Pose2(10.1, 27.1, 2.1));
  initialEstimate.print("\nInitial Estimate:\n"); // print

  // 4. Optimize using Levenberg-Marquardt optimization. The optimizer
  // accepts an optional set of configuration parameters, controlling
  // things like convergence criteria, the type of linear system solver
  // to use, and the amount of information displayed during optimization.
  // Here we will use the default set of parameters.  See the
  // documentation for the full set of parameters.
  LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
  Values result = optimizer.optimize();
  result.print("Final Result:\n");

  // 5. Calculate and print marginal covariances for all variables
  Marginals marginals(graph, result);
  cout << "x1 covariance:\n" << marginals.marginalCovariance(1) << endl;
  cout << "x2 covariance:\n" << marginals.marginalCovariance(2) << endl;
  cout << "x3 covariance:\n" << marginals.marginalCovariance(3) << endl;

  return 0;
}
