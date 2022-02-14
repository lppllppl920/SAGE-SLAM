#ifndef DF_CODE_FACTOR_H_
#define DF_CODE_FACTOR_H_

#include <gtsam/linear/HessianFactor.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <cmath>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Dense>

#include "gtsam_traits.h"
#include "keyframe.h"

namespace df
{

  template <typename Scalar, int CS>
  class CodeFactor : public gtsam::NonlinearFactor
  {
    typedef CodeFactor<Scalar, CS> This;
    typedef gtsam::NonlinearFactor Base;
    typedef Keyframe<Scalar> KeyframeT;
    typedef typename KeyframeT::Ptr KeyframePtr;
    typedef gtsam::Vector CodeT;

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /*!
   * \brief Constructor calculating the sparse sampling
   */
    CodeFactor(const KeyframePtr &kf,
               const gtsam::Key &code_key,
               const CodeT &init_code,
               const Scalar &factor_weight);

    virtual ~CodeFactor();

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
    size_t dim() const override { return CS; }

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
      return shared_ptr(new This(kf_, code_key_, init_code_, factor_weight_));
    }

  private:
    double ComputeError(const CodeT &code) const;

    /* variables we tie with this factor */
    gtsam::Key code_key_;
    CodeT init_code_;
    /* we do modify the keyframes in the factor */
    KeyframePtr kf_;
    Scalar factor_weight_;

    mutable double error_;
    mutable Eigen::Matrix<double, CS, CS> AtA_;
    mutable Eigen::Matrix<double, CS, 1> Atb_;
    
    
  };

} // namespace df

#endif // DF_SCALE_FACTOR_H_
