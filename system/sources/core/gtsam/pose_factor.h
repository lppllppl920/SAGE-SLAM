#ifndef DF_POSE_FACTOR_H_
#define DF_POSE_FACTOR_H_

#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <sophus/se3.hpp>
#include <torch/torch.h>
#include <cmath>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Dense>

#include "gtsam_traits.h"
#include "pinhole_camera.h"
#include "keyframe.h"
#include "mapping_utils.h"

namespace df
{

  template <typename Scalar>
  class PoseFactor : public gtsam::NonlinearFactor
  {
    typedef PoseFactor<Scalar> This;
    typedef Sophus::SE3<Scalar> PoseT;
    typedef gtsam::NonlinearFactor Base;
    typedef Keyframe<Scalar> KeyframeT;
    typedef typename KeyframeT::Ptr KeyframePtr;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PoseFactor(const KeyframePtr &kf,
               const gtsam::Key &pose_key,
               const PoseT &target_pose,
               const Scalar &factor_weight);

    virtual ~PoseFactor();

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
    size_t dim() const override { return 6; }

    /*!
   * \brief Return a string describing this factor
   * \return Factor description string
   */
    std::string Name() const;

    /*!
    * \brief Clone the factor
    * \return shared ptr to a cloned factor
    */
    virtual shared_ptr clone() const override
    {
      return shared_ptr(new This(kf_, pose_key_, target_pose_, factor_weight_));
    }

  private:
    Scalar ComputeError(const PoseT &pose) const;
    void ComputeJacobianAndError(const PoseT &pose) const;

    /* variables we tie with this factor */
    gtsam::Key pose_key_;

    /* we do modify the keyframes in the factor */
    KeyframePtr kf_;

    PoseT target_pose_;
    Scalar factor_weight_;

    mutable Eigen::Matrix<double, 6, 6> AtA_;
    mutable Eigen::Matrix<double, 6, 1> Atb_;
    mutable double error_;

  };

} // namespace df

#endif // DF_POSE_FACTOR_H_
