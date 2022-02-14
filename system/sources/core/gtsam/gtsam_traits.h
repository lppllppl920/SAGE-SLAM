#ifndef DF_GTSAM_TRAITS_H_
#define DF_GTSAM_TRAITS_H_

#include <sophus/se3.hpp>
#include <cmath>
#include <torch/torch.h>
#include <algorithm>
#include <Eigen/Geometry>

namespace gtsam
{

  template <typename Scalar>
  struct traits<Sophus::SE3<Scalar>>
  {
    typedef Sophus::SE3<Scalar> SE3T;

    /** Return the dimensionality of the tangent space of this value */
    static long GetDimension(const SE3T &pose)
    {
      return SE3T::DoF;
    }

    static void Print(const SE3T &pose, const std::string &str)
    {
      std::cout << str << pose.log().transpose();
    }

    /** Increment the value, by mapping3 from the vector delta in the tangent
    * space of the current value back to the manifold to produce a new,
    * incremented value.
    * @param delta The delta vector in the tangent space of this value, by
    * which to increment this value.
    */

    /*
   * valid expressions:
    * `size_t dim = traits<T>::GetDimension(p);` static function should be defined. This is mostly useful if the size is not known at compile time.
    * `v = traits<T>::Local(p,q)`, the chart, from manifold to tangent space, think of it as *q (-) p*, where *p* and *q* are elements of the manifold and the result, *v* is an element of the vector space.
    * `p = traits<T>::Retract(p,v)`, the inverse chart, from tangent space to manifold, think of it as *p (+) v*, where *p* is an element of the manifold and the result, *v* is an element of the vector space.
* invariants
    * `Retract(p, Local(p,q)) == q`
    * `Local(p, Retract(p, v)) == v`
   */
    static SE3T Retract(const SE3T &pose, const gtsam::Vector &delta)
    {
      // Eigen::Matrix<Scalar, 6, 1> update = delta.cast<Scalar>();
      // Eigen::Matrix<Scalar, 3, 1> trs_update = update.template head<3>();
      // Eigen::Matrix<Scalar, 3, 1> rot_update = update.template tail<3>();

      // SE3T pose_new;
      // pose_new.translation() = pose.translation() + trs_update;
      // pose_new.so3() = SE3T::SO3Type::exp(rot_update) * pose.so3();
      // return pose_new;

      Eigen::Matrix<Scalar, 6, 1> update = delta.cast<Scalar>();
      Eigen::Matrix<Scalar, 3, 1> trs_update = update.template head<3>();
      Eigen::Matrix<Scalar, 3, 1> rot_update = update.template tail<3>();
      Eigen::Matrix<Scalar, 3, 3> delta_rot_mat;
      Eigen::Matrix<Scalar, 3, 1> delta_trans_vec;
      se3_exp(rot_update, trs_update, delta_rot_mat, delta_trans_vec);
      SE3T pose_new;

      Eigen::AngleAxis<Scalar> angle_axis(delta_rot_mat * pose.so3().matrix());
      Eigen::Matrix<Scalar, 3, 3> temp = angle_axis.toRotationMatrix();

      pose_new.translation() = delta_rot_mat * pose.translation() + delta_trans_vec;
      pose_new.so3() = Sophus::SO3<Scalar>(temp);
      return pose_new;
    }

    /** Compute the coordinates in the tangent space of this value that
    * retract() would map to \c value.
    * @param value The value whose coordinates should be determined in the
    * tangent space of the value on which this function is called.
    * @return The coordinates of \c value in the tangent space of \c this.
    */
    static gtsam::Vector Local(const SE3T &origin, const SE3T &other)
    {
      // typename SE3T::Tangent tangent;
      // tangent.template head<3>() = other.translation() - origin.translation();
      // tangent.template tail<3>() = (other.so3() * origin.so3().inverse()).log();
      // return tangent.template cast<double>();
      // T1 * T0^(-1)
      typename SE3T::Tangent tangent;
      tangent.template head<3>() = other.translation() - other.so3() * origin.so3().inverse() * origin.translation();
      tangent.template tail<3>() = (other.so3() * origin.so3().inverse()).log();
      return tangent.template cast<double>();
    }

    /** Compare this Value with another for equality. */
    static bool Equals(const SE3T &first, const SE3T &second, Scalar tol)
    {
      return Local(first, second).norm() < tol;
    }

    static Eigen::Matrix<Scalar, 3, 3> so3_hat(const Eigen::Matrix<Scalar, 3, 1> &omega)
    {
      Eigen::Matrix<Scalar, 3, 3> omega_hat;
      omega_hat << 0.0, -omega(2, 0), omega(1, 0),
          omega(2, 0), 0.0, -omega(0, 0),
          -omega(1, 0), omega(0, 0), 0.0;

      return omega_hat;
    }

    static void se3_exp(const Eigen::Matrix<Scalar, 3, 1> &omega, const Eigen::Matrix<Scalar, 3, 1> &v,
                        Eigen::Matrix<Scalar, 3, 3> &R, Eigen::Matrix<Scalar, 3, 1> &t)
    {
      Scalar theta = omega.norm();
      Eigen::Matrix<Scalar, 3, 1> normalized_omega;
      if (theta > 0)
      {
        normalized_omega = omega / theta;
      }
      else
      {
        // when theta is zero, set a casual rotation direction vector
        normalized_omega << 1.0, 0.0, 0.0;
      }
      // TODO: hardcoded theta minimum value for now
      theta = std::max((Scalar)theta, (Scalar)1.0e-14);
      Scalar sintheta = std::sin(theta);
      Scalar costheta = std::cos(theta);

      Eigen::Matrix<Scalar, 3, 3> normalized_omega_hat = so3_hat(normalized_omega);
      Eigen::Matrix<Scalar, 3, 3> normalized_omega_hat_sq = normalized_omega_hat * normalized_omega_hat;

      Eigen::Matrix<Scalar, 3, 3> identity = Eigen::Matrix<Scalar, 3, 3>::Identity();

      R = identity + sintheta * normalized_omega_hat + (1.0 - costheta) * normalized_omega_hat_sq;

      Eigen::Matrix<Scalar, 3, 3> V = identity + ((1.0 - costheta) / theta) * normalized_omega_hat + ((theta - sintheta) / theta) * normalized_omega_hat_sq;
      t = V * v;

      return;
    }
  };

  /*
 * instantiate SE3 traits for double and float
 */
  template struct traits<Sophus::SE3<float>>;
  template struct traits<Sophus::SE3<double>>;

} // namespace gtsam

#endif // DF_GTSAM_TRAITS_H_
