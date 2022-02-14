#ifndef DF_REPROJECTION_FACTOR_H_
#define DF_REPROJECTION_FACTOR_H_

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
#include "reprojection_factor_kernels.h"

namespace df
{

  template <typename Scalar, int CS>
  class ReprojectionFactor : public gtsam::NonlinearFactor
  {
    typedef ReprojectionFactor<Scalar, CS> This;
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
    ReprojectionFactor(const PinholeCamera<Scalar> &cam,
                       const KeyframePtr &kf,
                       const FramePtr &fr,
                       const gtsam::Key &pose0_key,
                       const gtsam::Key &pose1_key,
                       const gtsam::Key &code0_key,
                       const gtsam::Key &scale0_key,
                       const long &num_keypoints,
                       const Scalar &cyc_consis_thresh,
                       const Scalar &factor_weight,
                       const Scalar &loss_param,
                       const Scalar &dpt_eps,
                       const teaser::RobustRegistrationSolver::Params &teaser_params,
                       const bool display_stats = false);

    virtual ~ReprojectionFactor();

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
    size_t dim() const override { return 13 + CS; }

    virtual shared_ptr clone() const
    {
      return shared_ptr(new This(cam_, kf_, fr_, pose0_key_, pose1_key_, code0_key_, scale0_key_,
                                 num_keypoints_, cyc_consis_thresh_, factor_weight_,
                                 loss_param_, dpt_eps_, teaser_params_));
    }

    /*!
   * \brief Return a string describing this factor
   * \return Factor description string
   */
    std::string Name() const;

    std::tuple<cv::Mat, cv::Mat> DrawMatches();
    cv::Mat ErrorImage() const;
    long InlierMatches() const { return num_inlier_matches_; }
    Scalar ReprojectionError() const { return error_; }
    FramePtr GetFrame() { return fr_; }
    KeyframePtr GetKeyframe() { return kf_; }

  private:
    Scalar ComputeError(const PoseT &pose0, const PoseT &pose1, const CodeT &code0, const Scalar &scale0) const;
    void ComputeJacobianAndError(const PoseT &pose0, const PoseT &pose1,
                                 const CodeT &code0, const Scalar &scale0) const;

    /* variables we tie with this factor */
    gtsam::Key pose0_key_;
    gtsam::Key pose1_key_;
    gtsam::Key code0_key_;
    gtsam::Key scale0_key_;

    PinholeCamera<Scalar> cam_;

    /* we do modify the keyframes in the factor */
    KeyframePtr kf_;
    FramePtr fr_;

    Scalar cyc_consis_thresh_;
    Scalar factor_weight_;
    Scalar loss_param_;
    Scalar inlier_multiplier_;
    Scalar desc_inlier_ratio_;

    Scalar dpt_eps_;

    teaser::RobustRegistrationSolver::Params teaser_params_;

    at::Tensor keypoint_indexes_;
    at::Tensor matched_keypoint_indexes_;

    at::Tensor matched_locations_1d_0_;
    at::Tensor matched_locations_homo_0_;
    at::Tensor matched_locations_2d_1_;

    long num_keypoints_;
    long num_inlier_matches_;

    /* system at the linearization point */
    mutable Eigen::Matrix<double, 13 + CS, 13 + CS> AtA_;
    mutable Eigen::Matrix<double, 13 + CS, 1> Atb_;

    mutable double error_;

    // debug purposes
    std::vector<cv::KeyPoint> keypoints_0_;
    std::vector<cv::KeyPoint> keypoints_1_;
    std::vector<cv::DMatch> desc_matches_;
    std::vector<cv::KeyPoint> display_corrs_1_;

    bool display_stats_;
  };

} // namespace df

#endif // DF_REPROJECTION_FACTOR_H_