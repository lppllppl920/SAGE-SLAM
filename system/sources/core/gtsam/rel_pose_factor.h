#ifndef DF_REL_POSE_FACTOR_H_
#define DF_REL_POSE_FACTOR_H_

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
  class RelPoseFactor : public gtsam::NonlinearFactor
  {
    typedef RelPoseFactor<Scalar> This;
    typedef Sophus::SE3<Scalar> PoseT;
    typedef gtsam::NonlinearFactor Base;
    typedef Keyframe<Scalar> KeyframeT;
    typedef typename KeyframeT::Ptr KeyframePtr;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RelPoseFactor(const KeyframePtr &kf0,
                  const KeyframePtr &kf1,
                  const gtsam::Key &pose0_key,
                  const gtsam::Key &pose1_key,
                  const PoseT &tgt_relpose10,
                  const Scalar &factor_weight,
                  const Scalar &rot_weight);

    virtual ~RelPoseFactor();

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
      return shared_ptr(new This(kf0_, kf1_, pose0_key_, pose1_key_, tgt_relpose10_, factor_weight_, rot_weight_));
    }

  private:
    Scalar ComputeError(const PoseT &pose0, const PoseT &pose1) const;
    void ComputeJacobianAndError(const PoseT &pose0, const PoseT &pose1) const;

    gtsam::Key pose0_key_;
    gtsam::Key pose1_key_;

    KeyframePtr kf0_;
    KeyframePtr kf1_;

    PoseT tgt_relpose10_;
    Scalar factor_weight_;
    Scalar rot_weight_;

    mutable double error_;
    mutable Eigen::Matrix<double, 12, 12> AtA_;
    mutable Eigen::Matrix<double, 12, 1> Atb_;


    Eigen::Matrix<Scalar, 9, 9> transpose_perm_mat_;
  };

} // namespace df

#endif // DF_REL_POSE_FACTOR_H_
