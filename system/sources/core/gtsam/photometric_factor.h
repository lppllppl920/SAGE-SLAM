#ifndef DF_PHOTOMETRIC_FACTOR_H_
#define DF_PHOTOMETRIC_FACTOR_H_

#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <sophus/se3.hpp>
#include <Eigen/Dense>
#include <torch/torch.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <sstream>
#include <limits>
#include <cmath>

#include "gtsam_traits.h"
#include "display_utils.h"
#include "pinhole_camera.h"
#include "keyframe.h"
#include "mapping_utils.h"
#include "photometric_factor_kernels.h"

namespace df
{
  template <typename Scalar, int CS>
  class PhotometricFactor : public gtsam::NonlinearFactor
  {
    typedef PhotometricFactor<Scalar, CS> This;
    typedef gtsam::NonlinearFactor Base;
    typedef Sophus::SE3<Scalar> PoseT;
    typedef gtsam::Vector CodeT;
    typedef Keyframe<Scalar> KeyframeT;
    typedef Frame<Scalar> FrameT;
    typedef typename KeyframeT::Ptr KeyframePtr;
    typedef typename FrameT::Ptr FramePtr;

  public:
    // This is to make the new operator aligned in terms of memory allocation, which we probably don't need it if we do not use custom CUDA operations.
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PhotometricFactor(
        const CameraPyramid<Scalar> &camera_pyramid,
        const KeyframePtr &kf,
        const FramePtr &fr,
        const gtsam::Key &pose0_key,
        const gtsam::Key &pose1_key,
        const gtsam::Key &code0_key,
        const gtsam::Key &scale0_key,
        const std::vector<Scalar> &factor_weights,
        const Scalar &dpt_eps,
        const bool display_stats=false);

    PhotometricFactor(
        const CameraPyramid<Scalar> &camera_pyramid,
        const KeyframePtr &kf,
        const FramePtr &fr,
        const gtsam::Key &pose0_key,
        const gtsam::Key &pose1_key,
        const gtsam::Key &code0_key,
        const gtsam::Key &scale0_key,
        const at::Tensor factor_weights,
        const Scalar &dpt_eps,
        const bool display_stats=false);

    virtual ~PhotometricFactor();

    /**
   * Calculate the error of the factor
   */
    // override the virtual function defined in the base class NonlinearFactor
    double error(const gtsam::Values &c) const override;

    /*
   * Linearize the factor around values c. Returns a HessianFactor
   */
    boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values &c) const override;

    /**
   * Get the dimension of the factor (number of rows on linearization)
   */
    size_t dim() const override { return 13 + CS; }

    virtual shared_ptr clone() const override
    {
      return shared_ptr(new This(camera_pyramid_, kf_, fr_, pose0_key_, pose1_key_, code0_key_, scale0_key_, factor_weights_, dpt_eps_, display_stats_));
    }

    /**
   * Return a string describing this factor
   */
    std::string Name() const;

  private:
    Scalar ComputeError(const PoseT &pose0, const PoseT &pose1, const CodeT &code0, const Scalar &scale0) const;
    void ComputeJacobianAndError(const PoseT &pose0, const PoseT &pose1,
                                 const CodeT &code0, const Scalar &scale0) const;

  private:
    /* variables we tie with this factor */
    gtsam::Key pose0_key_;
    gtsam::Key pose1_key_;
    gtsam::Key code0_key_;
    gtsam::Key scale0_key_;

    CameraPyramid<Scalar> camera_pyramid_;

    /* we do modify the keyframes in the factor */
    // the mutable keyword allows such variable to be changed even in a function declared with 'const' keyword
    KeyframePtr kf_;
    FramePtr fr_;

    Scalar dpt_eps_;

    bool display_stats_;

    /* system at the linearization point */
    mutable double error_;
    mutable Eigen::Matrix<double, 13 + CS, 13 + CS> AtA_;
    mutable Eigen::Matrix<double, 13 + CS, 1> Atb_;

    at::Tensor factor_weights_;
  };

} // namespace df

#endif // DF_PHOTOMETRIC_FACTOR_H_
