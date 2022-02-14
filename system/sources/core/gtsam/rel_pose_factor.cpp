#include "rel_pose_factor.h"

#include <Eigen/KroneckerProduct>

namespace df
{

  template <typename Scalar>
  RelPoseFactor<Scalar>::RelPoseFactor(const KeyframePtr &kf0,
                                       const KeyframePtr &kf1,
                                       const gtsam::Key &pose0_key,
                                       const gtsam::Key &pose1_key,
                                       const PoseT &tgt_relpose10,
                                       const Scalar &factor_weight,
                                       const Scalar &rot_weight)
      : Base(gtsam::cref_list_of<2>(pose0_key)(pose1_key)),
        pose0_key_(pose0_key), pose1_key_(pose1_key),
        kf0_(kf0), kf1_(kf1), tgt_relpose10_(tgt_relpose10),
        factor_weight_(factor_weight), rot_weight_(rot_weight), error_(0.0)
  {
    AtA_.setZero();
    Atb_.setZero();
    
    transpose_perm_mat_ = Eigen::Matrix<Scalar, 9, 9>::Zero();
    transpose_perm_mat_(0, 0) = 1;
    transpose_perm_mat_(1, 3) = 1;
    transpose_perm_mat_(2, 6) = 1;
    transpose_perm_mat_(3, 1) = 1;
    transpose_perm_mat_(4, 4) = 1;
    transpose_perm_mat_(5, 7) = 1;
    transpose_perm_mat_(6, 2) = 1;
    transpose_perm_mat_(7, 5) = 1;
    transpose_perm_mat_(8, 8) = 1;
  }

  /* ************************************************************************* */
  template <typename Scalar>
  RelPoseFactor<Scalar>::~RelPoseFactor() {}

  /* ************************************************************************* */
  template <typename Scalar>
  double RelPoseFactor<Scalar>::error(const gtsam::Values &c) const
  {
    if (this->active(c))
    {
      // get values of the optimization variables
      PoseT p0 = c.at<PoseT>(pose0_key_);
      PoseT p1 = c.at<PoseT>(pose1_key_);
      Scalar error = ComputeError(p0, p1);
      return (double)error;
    }
    else
    {
      return 0.0;
    }
  }

  /* ************************************************************************* */
  template <typename Scalar>
  boost::shared_ptr<gtsam::GaussianFactor>
  RelPoseFactor<Scalar>::linearize(const gtsam::Values &c) const
  {
    // Only linearize if the factor is active
    if (!this->active(c))
    {
      return boost::shared_ptr<gtsam::HessianFactor>();
    }

    // recover our values
    PoseT p0 = c.at<PoseT>(pose0_key_);
    PoseT p1 = c.at<PoseT>(pose1_key_);
    ComputeJacobianAndError(p0, p1);

    // Eigen::Matrix<double, 12, 12> oriAtA;
    // Eigen::Matrix<double, 12, 1> oriAtb;
    // double orierror;
    // oriAtA = AtA_;
    // oriAtb = Atb_;
    // orierror = error_;

    // Scalar eps = 1.0e-6;
    // PoseT modified_p1 = p1;
    // modified_p1.translation() = modified_p1.translation().array() + eps;

    // ComputeJacobianAndError(p0, modified_p1);
    // // (err(x+\deltax) - err(x)) = \deltax^T * cuAtA * \deltax + 2 * cuAtb^T * \deltax
    // double d_err_numeric = (error_ - orierror);
    // Eigen::Matrix<double, 12, 1> delta_x = Eigen::Matrix<double, 12, 1>::Zero();
    // delta_x(6, 0) = eps;
    // delta_x(7, 0) = eps;
    // delta_x(8, 0) = eps;
    // auto d_err_analytic = delta_x.transpose() * oriAtA * delta_x - 2.0 * oriAtb.transpose() * delta_x;
    // LOG(INFO) << "[RelPoseFactor<Scalar>::linearize] d_err numeric: " << d_err_numeric << " d_err analytic: " << d_err_analytic;

    
    const gtsam::FastVector<gtsam::Key> keys = {pose0_key_, pose1_key_};

    VLOG(3) << "-----------------------------------";
    VLOG(3) << "[RelPoseFactor<Scalar>::linearize] Asking to linearize " << Name() << " at values:";
    VLOG(3) << "[RelPoseFactor<Scalar>::linearize] curr pose0: " << p0.log().transpose() << " pose1: " << p1.log().transpose();
    VLOG(3) << "[RelPoseFactor<Scalar>::linearize] curr relative pose: " << (p1.inverse() * p0).log().transpose();
    VLOG(3) << "[RelPoseFactor<Scalar>::linearize] tgt relative pose: " << tgt_relpose10_.log().transpose();
    VLOG(3) << "[RelPoseFactor<Scalar>::linearize] error: " << error_;
    VLOG(3) << "-----------------------------------";

    std::vector<gtsam::Matrix> Gs;
    std::vector<gtsam::Vector> gs;

    Eigen::MatrixXd M = AtA_.template cast<double>();
    Eigen::MatrixXd corrected_AtA = NearestPsd(M);

    /*
     * Hessian composition
     *
     *      p0   p1
     * p0 [ G11 G12 ]
     * p1 [     G22 ]
     */
    const Eigen::MatrixXd G11 = corrected_AtA.template block<6, 6>(0, 0);
    const Eigen::MatrixXd G12 = corrected_AtA.template block<6, 6>(0, 6);
    const Eigen::MatrixXd G22 = corrected_AtA.template block<6, 6>(6, 6);

    Gs.push_back(G11);
    Gs.push_back(G12);
    Gs.push_back(G22);

    /*
    * Jtr composition
    *
    * p0 [ g1 ]
    * p1 [ g2 ]
    */
    const Eigen::MatrixXd g1 = Atb_.template block<6, 1>(0, 0);
    const Eigen::MatrixXd g2 = Atb_.template block<6, 1>(6, 0);

    gs.push_back(g1);
    gs.push_back(g2);

    // create and return HessianFactor
    return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)error_);
  }

  /* ************************************************************************* */
  template <typename Scalar>
  std::string RelPoseFactor<Scalar>::Name() const
  {
    std::stringstream ss;
    auto fmt = gtsam::DefaultKeyFormatter;
    ss << "RelPoseFactor " << kf0_->id << " -> " << kf1_->id << " keys = {" << fmt(pose0_key_) << ", " << fmt(pose1_key_) << "}";
    return ss.str();
  }

  /* ************************************************************************* */
  template <typename Scalar>
  inline Scalar RelPoseFactor<Scalar>::ComputeError(const PoseT &pose0, const PoseT &pose1) const
  {
    const PoseT relpose10 = pose1.inverse() * pose0;
    return factor_weight_ * (rot_weight_ * (relpose10.so3().log().array() - tgt_relpose10_.so3().log().array()).square().sum() +
                             (relpose10.translation().array() - tgt_relpose10_.translation().array()).square().sum());
  }

  /* ************************************************************************* */
  template <typename Scalar>
  inline void RelPoseFactor<Scalar>::ComputeJacobianAndError(const PoseT &pose0, const PoseT &pose1) const
  {
    const PoseT relpose10 = pose1.inverse() * pose0;

    Scalar rot_error = rot_weight_ * (relpose10.so3().log().array() - tgt_relpose10_.so3().log().array()).square().sum();
    Scalar trans_error = (relpose10.translation().array() - tgt_relpose10_.translation().array()).square().sum();
    error_ = factor_weight_ * (rot_error + trans_error);

    VLOG(3) << "[RelPoseFactor<Scalar>::ComputeJacobianAndError] rot and trans error for kf : " << kf0_->id << " and " << kf1_->id << " : " << rot_error << " " << trans_error;

    Eigen::Matrix<Scalar, 3, 3> R10 = relpose10.so3().matrix();
    Scalar trace_R10 = R10.trace();
    Scalar costheta = (trace_R10 - 1.0) / 2.0;
    Scalar theta = std::acos(costheta);
    Scalar sintheta = sqrt(1.0 - costheta * costheta);
    Eigen::Matrix<Scalar, 3, 1> a;
    a << R10(2, 1) - R10(1, 2), R10(0, 2) - R10(2, 0), R10(1, 0) - R10(0, 1);
    a = a * (theta * costheta - sintheta) / (4 * std::pow(sintheta, 3));
    Scalar b = theta / (2.0 * sintheta);

    Eigen::Matrix<Scalar, 3, 9> dlogR_dR = Eigen::Matrix<Scalar, 3, 9>::Zero();
    if ((1.0 - costheta) < 1.0e-14)
    {
      dlogR_dR(0, 5) = 0.5;
      dlogR_dR(0, 7) = -0.5;
      dlogR_dR(1, 2) = -0.5;
      dlogR_dR(1, 6) = 0.5;
      dlogR_dR(2, 1) = 0.5;
      dlogR_dR(2, 3) = -0.5;
    }
    else
    {
      dlogR_dR(0, 0) = a(0, 0);
      dlogR_dR(1, 0) = a(1, 0);
      dlogR_dR(2, 0) = a(2, 0);
      dlogR_dR(2, 1) = b;
      dlogR_dR(1, 2) = -b;
      dlogR_dR(2, 3) = -b;
      dlogR_dR(0, 4) = a(0, 0);
      dlogR_dR(1, 4) = a(1, 0);
      dlogR_dR(2, 4) = a(2, 0);
      dlogR_dR(0, 5) = b;
      dlogR_dR(1, 6) = b;
      dlogR_dR(0, 7) = -b;
      dlogR_dR(0, 8) = a(0, 0);
      dlogR_dR(1, 8) = a(1, 0);
      dlogR_dR(2, 8) = a(2, 0);
    }

    Eigen::Matrix<Scalar, 6, 12> pseudo_dlog_dpose10 = Eigen::Matrix<Scalar, 6, 12>::Zero();
    pseudo_dlog_dpose10.block(0, 9, 3, 3) = Eigen::Matrix<Scalar, 3, 3>::Identity();
    pseudo_dlog_dpose10.block(3, 0, 3, 9) = dlogR_dR;

    Eigen::Matrix<Scalar, 12, 12> dpose10_dpose0 = Eigen::kroneckerProduct(Eigen::Matrix<Scalar, 4, 4>::Identity(), pose1.so3().inverse().matrix());

    Eigen::Matrix<Scalar, 12, 6> dpose0_deps0 = Eigen::Matrix<Scalar, 12, 6>::Zero();
    Eigen::Matrix<Scalar, 3, 3> R0 = pose0.so3().matrix();
    dpose0_deps0.block(0, 3, 3, 3) = Sophus::SO3<Scalar>::hat(-R0.col(0));
    dpose0_deps0.block(3, 3, 3, 3) = Sophus::SO3<Scalar>::hat(-R0.col(1));
    dpose0_deps0.block(6, 3, 3, 3) = Sophus::SO3<Scalar>::hat(-R0.col(2));
    dpose0_deps0.block(9, 0, 3, 3) = Eigen::Matrix<Scalar, 3, 3>::Identity();
    dpose0_deps0.block(9, 3, 3, 3) = Sophus::SO3<Scalar>::hat(-pose0.translation());

    Eigen::Matrix<Scalar, 6, 6> pseudo_dlog_deps0 = pseudo_dlog_dpose10 * dpose10_dpose0 * dpose0_deps0;

    Eigen::Matrix<Scalar, 12, 12> dpose10_dpose1inv = Eigen::kroneckerProduct(pose0.matrix(), Eigen::Matrix<Scalar, 3, 3>::Identity());

    Eigen::Matrix<Scalar, 12, 12> dpose1inv_dpose1 = Eigen::Matrix<Scalar, 12, 12>::Zero();
    dpose1inv_dpose1.block(0, 0, 9, 9) = transpose_perm_mat_;
    dpose1inv_dpose1.block(9, 0, 3, 9) = Eigen::kroneckerProduct(Eigen::Matrix<Scalar, 3, 3>::Identity(), -pose1.translation().transpose());
    dpose1inv_dpose1.block(9, 9, 3, 3) = -pose1.so3().inverse().matrix();
    

    Eigen::Matrix<Scalar, 12, 6> dpose1_deps1 = Eigen::Matrix<Scalar, 12, 6>::Zero();
    Eigen::Matrix<Scalar, 3, 3> R1 = pose1.so3().matrix();
    dpose1_deps1.block(0, 3, 3, 3) = Sophus::SO3<Scalar>::hat(-R1.col(0));
    dpose1_deps1.block(3, 3, 3, 3) = Sophus::SO3<Scalar>::hat(-R1.col(1));
    dpose1_deps1.block(6, 3, 3, 3) = Sophus::SO3<Scalar>::hat(-R1.col(2));
    dpose1_deps1.block(9, 0, 3, 3) = Eigen::Matrix<Scalar, 3, 3>::Identity();
    dpose1_deps1.block(9, 3, 3, 3) = Sophus::SO3<Scalar>::hat(-pose1.translation());
    Eigen::Matrix<Scalar, 6, 6> pseudo_dlog_deps1 = pseudo_dlog_dpose10 * dpose10_dpose1inv * dpose1inv_dpose1 * dpose1_deps1;

    Eigen::Matrix<Scalar, 6, 12> pseudo_log_jac_pose;
    pseudo_log_jac_pose.block(0, 0, 3, 6) = pseudo_dlog_deps0.block(0, 0, 3, 6);
    pseudo_log_jac_pose.block(3, 0, 3, 6) = sqrt(rot_weight_) * pseudo_dlog_deps0.block(3, 0, 3, 6);

    pseudo_log_jac_pose.block(0, 6, 3, 6) = pseudo_dlog_deps1.block(0, 0, 3, 6);
    pseudo_log_jac_pose.block(3, 6, 3, 6) = sqrt(rot_weight_) * pseudo_dlog_deps1.block(3, 0, 3, 6);

    Eigen::Matrix<Scalar, 6, 1> pseudo_cur_log_pose10;
    pseudo_cur_log_pose10 << relpose10.translation(), sqrt(rot_weight_) * relpose10.so3().log();

    Eigen::Matrix<Scalar, 6, 1> pseudo_tgt_log_pose10;
    pseudo_tgt_log_pose10 << tgt_relpose10_.translation(), sqrt(rot_weight_) * tgt_relpose10_.so3().log();

    Eigen::Matrix<Scalar, 6, 1> pseudo_log_diff = pseudo_tgt_log_pose10 - pseudo_cur_log_pose10;

    AtA_ = factor_weight_ * (pseudo_log_jac_pose.transpose() * pseudo_log_jac_pose).template cast<double>();
    Atb_ = factor_weight_ * (pseudo_log_jac_pose.transpose() * pseudo_log_diff).template cast<double>();

    return;
  }

  /* ************************************************************************* */
  // explicit instantiation
  template class RelPoseFactor<float>;

} // namespace df
