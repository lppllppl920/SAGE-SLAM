#ifndef DF_MATCH_GEOMETRY_FACTOR_H_
#define DF_MATCH_GEOMETRY_FACTOR_H_

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

  template <typename Scalar, int CS>
  class MatchGeometryFactor : public gtsam::NonlinearFactor
  {
    typedef MatchGeometryFactor<Scalar, CS> This;
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
    MatchGeometryFactor(const PinholeCamera<Scalar> &cam,
                        const KeyframePtr &kf0,
                        const KeyframePtr &kf1,
                        const gtsam::Key &pose0_key,
                        const gtsam::Key &pose1_key,
                        const gtsam::Key &code0_key,
                        const gtsam::Key &code1_key,
                        const gtsam::Key &scale0_key,
                        const gtsam::Key &scale1_key,
                        const long &num_keypoints,
                        const Scalar &cyc_consis_thresh,
                        const Scalar &factor_weight,
                        const Scalar &loss_param,
                        const Scalar &dpt_eps,
                        const std::string robust_loss_type,
                        const teaser::RobustRegistrationSolver::Params &teaser_params,
                        const bool display_stats = false);

    virtual ~MatchGeometryFactor();

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
    size_t dim() const override { return 14 + 2 * CS; }

    virtual shared_ptr clone() const override
    {
      return shared_ptr(new This(cam_, kf0_, kf1_, pose0_key_, pose1_key_, code0_key_, code1_key_,
                                 scale0_key_, scale1_key_, num_keypoints_, cyc_consis_thresh_,
                                 factor_weight_, loss_param_, dpt_eps_, robust_loss_type_, teaser_params_));
    }

    /*!
   * \brief Return a string describing this factor
   * \return Factor description string
   */
    std::string Name() const;

    std::tuple<cv::Mat, cv::Mat> DrawMatches();
    cv::Mat ErrorImage() const;
    long InlierMatches() const { return num_inlier_matches_; }
    Scalar DescInlierRatio() const { return desc_inlier_ratio_; }
    Scalar Error() const { return error_; }
    KeyframePtr GetKeyframe(const int idx) const
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

    const at::Tensor GetMatchLocations1D(const int idx) const
    {
      switch (idx)
      {
      case 0:
        return matched_locations_1d_0_.to(torch::kLong);
      case 1:
        return matched_locations_1d_1_.to(torch::kLong);
      default:
        LOG(FATAL) << "[MatchGeometryFactor::GetMatchLocations1D] index " << idx << " not supported";
      }
    }

    const at::Tensor GetMatchLocationsHomo(const int idx) const
    {
      switch (idx)
      {
      case 0:
        return matched_locations_homo_0_;
      case 1:
        return matched_locations_homo_1_;
      default:
        LOG(FATAL) << "[MatchGeometryFactor::GetMatchLocationsHomo] index " << idx << " not supported";
      }
    }

    Scalar GetLossParam()
    {
      return loss_param_;
    }

  private:
    Scalar ComputeError(const PoseT &pose0, const PoseT &pose1, const CodeT &code0,
                        const CodeT &code1, const Scalar &scale0, const Scalar &scale1) const;
    void ComputeJacobianAndError(const PoseT &pose0, const PoseT &pose1, const CodeT &code0,
                                 const CodeT &code1, const Scalar &scale0, const Scalar &scale1) const;

    /* variables we tie with this factor */
    gtsam::Key pose0_key_;
    gtsam::Key pose1_key_;
    gtsam::Key code0_key_;
    gtsam::Key code1_key_;
    gtsam::Key scale0_key_;
    gtsam::Key scale1_key_;

    PinholeCamera<Scalar> cam_;

    /* we do modify the keyframes in the factor */
    KeyframePtr kf0_;
    KeyframePtr kf1_;

    Scalar cyc_consis_thresh_;
    Scalar factor_weight_;
    Scalar loss_param_;

    Scalar dpt_eps_;

    bool display_stats_;

    std::string robust_loss_type_;

    at::Tensor keypoint_indexes_;
    at::Tensor matched_keypoint_indexes_;

    at::Tensor matched_locations_1d_0_;
    at::Tensor matched_locations_homo_0_;

    at::Tensor matched_locations_1d_1_;
    at::Tensor matched_locations_homo_1_;

    long num_keypoints_;
    long num_inlier_matches_;

    // debug purposes
    std::vector<cv::KeyPoint> keypoints_0_;
    std::vector<cv::KeyPoint> keypoints_1_;
    std::vector<cv::DMatch> desc_matches_;
    std::vector<cv::KeyPoint> display_corrs_1_;

    Scalar desc_inlier_ratio_;
    Scalar inlier_multiplier_;

    teaser::RobustRegistrationSolver::Params teaser_params_;

    /* system at the linearization point */
    mutable double error_;
    mutable Eigen::Matrix<double, 14 + 2 * CS, 14 + 2 * CS, Eigen::DontAlign> AtA_;
    mutable Eigen::Matrix<double, 14 + 2 * CS, 1, Eigen::DontAlign> Atb_;
  };

} // namespace df

#endif // DF_MATCH_GEOMETRY_FACTOR_H_
