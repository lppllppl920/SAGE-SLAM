/*
 * @file OrientedPlane3Factor.cpp
 * @brief OrientedPlane3 Factor class
 * @author Alex Trevor
 * @date December 22, 2013
 */

#pragma once

#include <gtsam/geometry/OrientedPlane3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace gtsam {

/**
 * Factor to measure a planar landmark from a given pose
 */
class OrientedPlane3Factor: public NoiseModelFactor2<Pose3, OrientedPlane3> {

protected:
  Key poseKey_;
  Key landmarkKey_;
  Vector measured_coeffs_;
  OrientedPlane3 measured_p_;

  typedef NoiseModelFactor2<Pose3, OrientedPlane3> Base;

public:

  /// Constructor
  OrientedPlane3Factor() {
  }
  virtual ~OrientedPlane3Factor() {}

  /// Constructor with measured plane coefficients (a,b,c,d), noise model, pose symbol
  OrientedPlane3Factor(const Vector&z, const SharedGaussian& noiseModel,
      const Key& pose, const Key& landmark) :
      Base(noiseModel, pose, landmark), poseKey_(pose), landmarkKey_(landmark), measured_coeffs_(
          z) {
    measured_p_ = OrientedPlane3(Unit3(z(0), z(1), z(2)), z(3));
  }

  /// print
  virtual void print(const std::string& s = "OrientedPlane3Factor",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const;

  /// evaluateError
  virtual Vector evaluateError(const Pose3& pose, const OrientedPlane3& plane,
      boost::optional<Matrix&> H1 = boost::none, boost::optional<Matrix&> H2 =
          boost::none) const {
    OrientedPlane3 predicted_plane = OrientedPlane3::Transform(plane, pose, H1,
        H2);
    Vector err(3);
    err << predicted_plane.error(measured_p_);
    return (err);
  }
  ;
};

// TODO: Convert this factor to dimension two, three dimensions is redundant for direction prior
class OrientedPlane3DirectionPrior: public NoiseModelFactor1<OrientedPlane3> {
protected:
  OrientedPlane3 measured_p_; /// measured plane parameters
  Key landmarkKey_;
  typedef NoiseModelFactor1<OrientedPlane3> Base;
public:

  typedef OrientedPlane3DirectionPrior This;
  /// Constructor
  OrientedPlane3DirectionPrior() {
  }

  /// Constructor with measured plane coefficients (a,b,c,d), noise model, landmark symbol
  OrientedPlane3DirectionPrior(Key key, const Vector&z,
      const SharedGaussian& noiseModel) :
      Base(noiseModel, key), landmarkKey_(key) {
    measured_p_ = OrientedPlane3(Unit3(z(0), z(1), z(2)), z(3));
  }

  /// print
  virtual void print(const std::string& s = "OrientedPlane3DirectionPrior",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const;

  /// equals
  virtual bool equals(const NonlinearFactor& expected, double tol = 1e-9) const;

  virtual Vector evaluateError(const OrientedPlane3& plane,
      boost::optional<Matrix&> H = boost::none) const;
};

} // gtsam

