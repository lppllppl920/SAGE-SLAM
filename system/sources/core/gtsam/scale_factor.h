#ifndef DF_SCALE_FACTOR_H_
#define DF_SCALE_FACTOR_H_

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
  class ScaleFactor : public gtsam::NonlinearFactor
  {
    typedef ScaleFactor<Scalar> This;
    typedef gtsam::NonlinearFactor Base;
    typedef Keyframe<Scalar> KeyframeT;
    typedef typename KeyframeT::Ptr KeyframePtr;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /*!
   * \brief Constructor calculating the sparse sampling
   */
    ScaleFactor(const KeyframePtr &kf,
                const gtsam::Key &scale_key,
                const Scalar &init_scale,
                const Scalar &factor_weight);

    // /*!
    //  * \brief Constructor that takes sampled points (for clone)
    //  */
    // ScaleFactor(const PinholeCamera<Scalar>& cam,
    //                       const KeyframePtr& kf0,
    //                       const KeyframePtr& kf1,
    //                       const gtsam::Key& pose0_key,
    //                       const gtsam::Key& pose1_key,
    //                       const gtsam::Key& code0_key,
    //                       const gtsam::Key& code1_key,;

    virtual ~ScaleFactor();

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
    size_t dim() const override { return 1; }

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
      return shared_ptr(new This(kf_, scale_key_, init_scale_, factor_weight_));
    }

  private:
    Scalar ComputeError(const Scalar &scale) const;
    void ComputeJacobianAndError(const Scalar &scale) const;

    /* variables we tie with this factor */
    gtsam::Key scale_key_;

    /* we do modify the keyframes in the factor */
    KeyframePtr kf_;

    Scalar init_scale_;
    Scalar factor_weight_;

    mutable double error_;
    mutable Eigen::Matrix<double, 1, 1> AtA_, Atb_;
  };

} // namespace df

#endif // DF_SCALE_FACTOR_H_
