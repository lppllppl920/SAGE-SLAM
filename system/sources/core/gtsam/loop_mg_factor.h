#ifndef DF_LOOP_MG_FACTOR_H_
#define DF_LOOP_MG_FACTOR_H_

#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <sophus/se3.hpp>
#include <torch/torch.h>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <teaser/registration.h>

#include "gtsam_traits.h"
#include "pinhole_camera.h"
#include "keyframe.h"
#include "mapping_utils.h"
#include "match_geometry_factor_kernels.h"

namespace df
{

  template <typename Scalar>
  class LoopMGFactor : public gtsam::NonlinearFactor
  {
    typedef LoopMGFactor<Scalar> This;
    typedef gtsam::NonlinearFactor Base;
    typedef Sophus::SE3<Scalar> PoseT;
    typedef gtsam::Vector CodeT;
    typedef Keyframe<Scalar> KeyframeT;
    typedef Frame<Scalar> FrameT;
    typedef typename KeyframeT::Ptr KeyframePtr;
    typedef typename FrameT::Ptr FramePtr;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /*!
   * \brief Constructor calculating the keypoint matches between kf0 and kf1
   */
    LoopMGFactor(const KeyframePtr &kf0,
                 const KeyframePtr &kf1,
                 const gtsam::Key &pose0_key,
                 const gtsam::Key &pose1_key,
                 const gtsam::Key &scale0_key,
                 const gtsam::Key &scale1_key,
                 const at::Tensor matched_unscaled_dpts_0,
                 const at::Tensor matched_unscaled_dpts_1,
                 const at::Tensor matched_locations_homo_0,
                 const at::Tensor matched_locations_homo_1,
                 const Scalar &factor_weight,
                 const Scalar &loss_param,
                 const Scalar &dpt_eps);

    virtual ~LoopMGFactor();

    /*!
   * \brief Calculate the error of the factor
   * \param c Values to evaluate the error at
   * \return The error
   */
    double error(const gtsam::Values &c) const override;

    /*!
   * \brief Linearizes this factor at a linearization point
   * \param c Linearization point
   * \return Linearized factor
   */
    boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values &c) const override;

    //   /*!
    //  * \brief Get the dimension of the factor (number of rows on linearization)
    //  * \return
    //  */
    size_t dim() const override { return 14; }

    virtual shared_ptr clone() const override
    {
      return shared_ptr(new This(kf0_, kf1_, pose0_key_, pose1_key_,
                                 scale0_key_, scale1_key_,
                                 matched_unscaled_dpts_0_, matched_unscaled_dpts_1_,
                                 matched_locations_homo_0_, matched_locations_homo_1_,
                                 factor_weight_, loss_param_, dpt_eps_));
    }

    /*!
   * \brief Return a string describing this factor
   * \return Factor description string
   */
    std::string Name() const;

    KeyframePtr GetKeyframe(const int idx)
    {
      switch (idx)
      {
      case 0:
        return kf0_;
      case 1:
        return kf1_;
      default:
        return nullptr;
      }
    }

  private:
    Scalar ComputeError(const PoseT &pose0, const PoseT &pose1, const Scalar &scale0, const Scalar &scale1) const;
    void ComputeJacobianAndError(const PoseT &pose0, const PoseT &pose1, const Scalar &scale0, const Scalar &scale1) const;

    /* variables we tie with this factor */
    gtsam::Key pose0_key_;
    gtsam::Key pose1_key_;
    gtsam::Key scale0_key_;
    gtsam::Key scale1_key_;

    /* we do modify the keyframes in the factor */
    KeyframePtr kf0_;
    KeyframePtr kf1_;

    Scalar factor_weight_;
    Scalar loss_param_;
    Scalar dpt_eps_;

    at::Tensor matched_unscaled_dpts_0_;
    at::Tensor matched_unscaled_dpts_1_;

    at::Tensor matched_locations_homo_0_;
    at::Tensor matched_locations_homo_1_;

    /* system at the linearization point */
    mutable double error_;
    mutable Eigen::Matrix<double, 14, 14, Eigen::DontAlign> AtA_;
    mutable Eigen::Matrix<double, 14, 1, Eigen::DontAlign> Atb_;
  };

} // namespace df

#endif // DF_LOOP_MG_FACTOR_H_
