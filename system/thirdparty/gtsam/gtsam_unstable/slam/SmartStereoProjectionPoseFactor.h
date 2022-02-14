/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   SmartStereoProjectionPoseFactor.h
 * @brief  Smart stereo factor on poses, assuming camera calibration is fixed
 * @author Luca Carlone
 * @author Chris Beall
 * @author Zsolt Kira
 * @author Frank Dellaert
 */

#pragma once

#include <gtsam_unstable/slam/SmartStereoProjectionFactor.h>

namespace gtsam {
/**
 *
 * @addtogroup SLAM
 *
 * If you are using the factor, please cite:
 * L. Carlone, Z. Kira, C. Beall, V. Indelman, F. Dellaert, Eliminating conditionally
 * independent sets in factor graphs: a unifying perspective based on smart factors,
 * Int. Conf. on Robotics and Automation (ICRA), 2014.
 *
 */

/**
 * This factor assumes that camera calibration is fixed, but each camera
 * has its own calibration.
 * The factor only constrains poses (variable dimension is 6).
 * This factor requires that values contains the involved poses (Pose3).
 * @addtogroup SLAM
 */
class SmartStereoProjectionPoseFactor: public SmartStereoProjectionFactor {

protected:

  std::vector<boost::shared_ptr<Cal3_S2Stereo> > K_all_; ///< shared pointer to calibration object (one for each camera)

public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// shorthand for base class type
  typedef SmartStereoProjectionFactor Base;

  /// shorthand for this class
  typedef SmartStereoProjectionPoseFactor This;

  /// shorthand for a smart pointer to a factor
  typedef boost::shared_ptr<This> shared_ptr;

  /**
   * Constructor
   * @param Isotropic measurement noise
   * @param params internal parameters of the smart factors
   */
  SmartStereoProjectionPoseFactor(const SharedNoiseModel& sharedNoiseModel,
      const SmartStereoProjectionParams& params = SmartStereoProjectionParams(),
      const boost::optional<Pose3> body_P_sensor = boost::none) :
      Base(sharedNoiseModel, params, body_P_sensor) {
  }

  /** Virtual destructor */
  virtual ~SmartStereoProjectionPoseFactor() {}

  /**
   * add a new measurement and pose key
   * @param measured is the 2m dimensional location of the projection of a single landmark in the m view (the measurement)
   * @param poseKey is key corresponding to the camera observing the same landmark
   * @param K is the (fixed) camera calibration
   */
  void add(const StereoPoint2 measured, const Key poseKey,
      const boost::shared_ptr<Cal3_S2Stereo> K) {
    Base::add(measured, poseKey);
    K_all_.push_back(K);
  }

  /**
   *  Variant of the previous one in which we include a set of measurements
   * @param measurements vector of the 2m dimensional location of the projection of a single landmark in the m view (the measurement)
   * @param poseKeys vector of keys corresponding to the camera observing the same landmark
   * @param Ks vector of calibration objects
   */
  void add(std::vector<StereoPoint2> measurements, KeyVector poseKeys,
      std::vector<boost::shared_ptr<Cal3_S2Stereo> > Ks) {
    Base::add(measurements, poseKeys);
    for (size_t i = 0; i < measurements.size(); i++) {
      K_all_.push_back(Ks.at(i));
    }
  }

  /**
   * Variant of the previous one in which we include a set of measurements with the same noise and calibration
   * @param measurements vector of the 2m dimensional location of the projection of a single landmark in the m view (the measurement)
   * @param poseKeys vector of keys corresponding to the camera observing the same landmark
   * @param K the (known) camera calibration (same for all measurements)
   */
  void add(std::vector<StereoPoint2> measurements, KeyVector poseKeys,
      const boost::shared_ptr<Cal3_S2Stereo> K) {
    for (size_t i = 0; i < measurements.size(); i++) {
      Base::add(measurements.at(i), poseKeys.at(i));
      K_all_.push_back(K);
    }
  }

  /**
   * print
   * @param s optional string naming the factor
   * @param keyFormatter optional formatter useful for printing Symbols
   */
  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
      DefaultKeyFormatter) const {
    std::cout << s << "SmartStereoProjectionPoseFactor, z = \n ";
    for(const boost::shared_ptr<Cal3_S2Stereo>& K: K_all_)
    K->print("calibration = ");
    Base::print("", keyFormatter);
  }

  /// equals
  virtual bool equals(const NonlinearFactor& p, double tol = 1e-9) const {
    const SmartStereoProjectionPoseFactor *e =
        dynamic_cast<const SmartStereoProjectionPoseFactor*>(&p);

    return e && Base::equals(p, tol);
  }

  /**
   * error calculates the error of the factor.
   */
  virtual double error(const Values& values) const {
    if (this->active(values)) {
      return this->totalReprojectionError(cameras(values));
    } else { // else of active flag
      return 0.0;
    }
  }

  /** return the calibration object */
  inline const std::vector<boost::shared_ptr<Cal3_S2Stereo> > calibration() const {
    return K_all_;
  }

  /**
   * Collect all cameras involved in this factor
   * @param values Values structure which must contain camera poses corresponding
   * to keys involved in this factor
   * @return vector of Values
   */
   Base::Cameras cameras(const Values& values) const {
    Base::Cameras cameras;
    size_t i=0;
    for(const Key& k: this->keys_) {
      Pose3 pose = values.at<Pose3>(k);

      if (Base::body_P_sensor_)
    	  pose = pose.compose(*(Base::body_P_sensor_));

      StereoCamera camera(pose, K_all_[i++]);
      cameras.push_back(camera);
    }
    return cameras;
  }

private:

  /// Serialization function
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
    ar & BOOST_SERIALIZATION_NVP(K_all_);
  }

}; // end of class declaration

/// traits
template<>
struct traits<SmartStereoProjectionPoseFactor> : public Testable<
    SmartStereoProjectionPoseFactor> {
};

} // \ namespace gtsam
