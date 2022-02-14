#ifndef DF_MAPPING_UTILS_H_
#define DF_MAPPING_UTILS_H_

#include <memory>
#include <Eigen/Dense>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/base/Vector.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <gtsam/base/cholesky.h>
#include <Eigen/KroneckerProduct>
#include <cmath>

#include "keyframe.h"
#include "display_utils.h"
#include "timing.h"
#include "keyframe_map.h"
#include "df_work.h"
#include "pinhole_camera.h"

namespace df
{

#define LIST(h, w) (std::initializer_list<TensorIndex>({h, w}))

  void PruneMatchesEightPoint(at::Tensor &inlier_indexes,
                              const at::Tensor keypoint_homo_locations_0,
                              const at::Tensor matched_homo_locations_1,
                              double threshold,
                              int max_iterations,
                              double probability);

  template <typename Scalar>
  cv::Mat DisplayKeyframes(typename Map<Scalar>::Ptr map, int num_keyframes);

  template <typename Scalar>
  cv::Mat DisplaySE3Warp(const Keyframe<Scalar> &keyframe_0, const Frame<Scalar> &frame_1,
                         const at::Tensor rotation_10, const at::Tensor translation_10, const Scalar &dpt_eps,
                         const cv::Mat &checkerboard);

  template <typename Scalar>
  cv::Mat DisplaySE3Warp(const df::Frame<Scalar> &frame_0, const df::Frame<Scalar> &frame_1,
                         const at::Tensor rotation_10, const at::Tensor translation_10, const float scale_0, const Scalar &dpt_eps,
                         const cv::Mat &checkerboard);

  template <typename Scalar>
  cv::Mat DisplayPairs(const typename Map<Scalar>::Ptr &map,
                       const std::vector<std::pair<long, long>> &pairs,
                       const at::Tensor video_mask,
                       const PinholeCamera<Scalar> &cam, const Scalar &dpt_eps,
                       int N, const cv::Mat &checkerboard);

  void GenerateCheckerboard(cv::Mat &checkboard, const std::vector<long> &image_size);

  void GenerateMaskPyramid(const at::Tensor valid_mask,
                           const int &num_levels,
                           const std::shared_ptr<std::vector<at::Tensor>> &valid_mask_pyramid_ptr);

  template <typename Scalar>
  inline void Locations3DNegDepthClamp(const at::Tensor locations_3d, const Scalar &eps, at::Tensor &valid_mask, at::Tensor &clamp_locations_3d)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    const at::Tensor dpt = locations_3d.index({Slice(), Slice(2, 3)});
    valid_mask = dpt > eps;
    clamp_locations_3d = torch::cat({locations_3d.index({Slice(), Slice(None, 2)}), torch::clamp_min(dpt, eps)}, 1);
    return;
  }

  template <typename Scalar>
  inline void Locations3DSmallDepthClamp(const at::Tensor locations_3d, const Scalar &eps, at::Tensor &valid_mask)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    const at::Tensor dpt = locations_3d.index({Slice(), Slice(2, 3)});
    valid_mask = torch::logical_or(dpt > eps, dpt < -eps);
    return;
  }

  template <typename T>
  inline bool IsPsd(const Eigen::MatrixBase<T> &M)
  {
    Eigen::LDLT<T> ldlt(M);
    return ldlt.isPositive();
  }

  /*
 * Find nearest positive semi-definite matrix to M
 *
 * [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
 *
 * [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
 * matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
 *
 */
  template <typename T>
  inline T NearestPsd(const Eigen::MatrixBase<T> &M)
  {
    typedef typename T::value_type Scalar;

    T B = (M + M.transpose()) / 2;

    Eigen::JacobiSVD<T> svd(B, Eigen::ComputeThinV);
    T H = svd.matrixV().transpose() * svd.singularValues().asDiagonal() * svd.matrixV();

    T A2 = (B + H) / 2;
    T A3 = (A2 + A2.transpose()) / 2;

    int k = 1;
    Scalar spacing = 1e-15;
    T I = T::Identity(M.rows(), M.cols());
    while (!IsPsd(A3))
    {
      Eigen::SelfAdjointEigenSolver<T> es(A3);
      Scalar min_eig = es.eigenvalues().minCoeff();
      A3 += I * (-min_eig * k + spacing);
      k *= 2;
    }
    return A3;
  }

  /**
 * Takes two poses in the same reference frame and
 * returns pose1 expressed in frame pose0
 * pose_ab = (pose_a)^-1 * pose_b
 */
  template <typename T, typename JacT = Eigen::Matrix<T, 6, 6>>
  inline void
  RelativePose(const Sophus::SE3<T> &pose_a, const Sophus::SE3<T> &pose_b, Sophus::SE3<T> &pose_ab)
  {
    pose_ab = pose_a.inverse() * pose_b;
    return;
  }

  inline at::Tensor RotationToQuaternion(const at::Tensor rotation_matrix, float eps)
  {
    /*        - Input: :math:`(3, 3)`
        - Output: :math:`(4,)`*/
    // 3 x 3
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    const at::Tensor rmat_t = rotation_matrix.permute({1, 0});
    const at::Tensor mask_d2 = rmat_t.index(LIST(2, 2)) < eps;
    const at::Tensor mask_d0_d1 = rmat_t.index(LIST(0, 0)) > rmat_t.index(LIST(1, 1));
    const at::Tensor mask_d0_nd1 = rmat_t.index(LIST(0, 0)) < -rmat_t.index(LIST(1, 1));
    const at::Tensor t0 = 1.0 + rmat_t.index(LIST(0, 0)) -
                          rmat_t.index(LIST(1, 1)) -
                          rmat_t.index(LIST(2, 2));
    const at::Tensor q0 = torch::stack({rmat_t.index(LIST(1, 2)) - rmat_t.index(LIST(2, 1)),
                                        t0, rmat_t.index(LIST(0, 1)) + rmat_t.index(LIST(1, 0)),
                                        rmat_t.index(LIST(2, 0)) + rmat_t.index(LIST(0, 2))},
                                       0);
    const at::Tensor t1 = 1.0 - rmat_t.index(LIST(0, 0)) + rmat_t.index(LIST(1, 1)) - rmat_t.index(LIST(2, 2));
    const at::Tensor q1 = torch::stack({rmat_t.index(LIST(2, 0)) - rmat_t.index(LIST(0, 2)),
                                        rmat_t.index(LIST(0, 1)) + rmat_t.index(LIST(1, 0)),
                                        t1, rmat_t.index(LIST(1, 2)) + rmat_t.index(LIST(2, 1))},
                                       0);

    const at::Tensor t2 = 1.0 - rmat_t.index(LIST(0, 0)) - rmat_t.index(LIST(1, 1)) + rmat_t.index(LIST(2, 2));
    const at::Tensor q2 = torch::stack({rmat_t.index(LIST(0, 1)) - rmat_t.index(LIST(1, 0)),
                                        rmat_t.index(LIST(2, 0)) + rmat_t.index(LIST(0, 2)),
                                        rmat_t.index(LIST(1, 2)) + rmat_t.index(LIST(2, 1)), t2},
                                       0);

    const at::Tensor t3 = 1.0 + rmat_t.index(LIST(0, 0)) + rmat_t.index(LIST(1, 1)) + rmat_t.index(LIST(2, 2));
    const at::Tensor q3 = torch::stack({t3, rmat_t.index(LIST(1, 2)) - rmat_t.index(LIST(2, 1)),
                                        rmat_t.index(LIST(2, 0)) - rmat_t.index(LIST(0, 2)),
                                        rmat_t.index(LIST(0, 1)) - rmat_t.index(LIST(1, 0))},
                                       0);

    const at::Tensor mask_c0 = mask_d2 * mask_d0_d1;
    const at::Tensor mask_c1 = mask_d2 * torch::logical_not(mask_d0_d1);
    const at::Tensor mask_c2 = torch::logical_not(mask_d2) * mask_d0_nd1;
    const at::Tensor mask_c3 = torch::logical_not(mask_d2) * torch::logical_not(mask_d0_nd1);
    // 4
    at::Tensor q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3;
    q = 0.5 * q / torch::sqrt(t0 * mask_c1 + t1 * mask_c1 + t2 * mask_c2 + t3 * mask_c3);

    return q;
  }

  inline at::Tensor QuaternionToAngleAxis(const at::Tensor quaternion)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    const at::Tensor q1 = quaternion.index({1});
    const at::Tensor q2 = quaternion.index({2});
    const at::Tensor q3 = quaternion.index({3});
    const at::Tensor sin_sq_theta = q1 * q1 + q2 * q2 + q3 * q3;
    const at::Tensor sin_theta = torch::sqrt(sin_sq_theta);
    const at::Tensor cos_theta = quaternion.index({0});

    const at::Tensor two_theta = cos_theta.item<float>() < 0.0 ? torch::atan2(-sin_theta, -cos_theta) : torch::atan2(sin_theta, cos_theta);
    const at::Tensor k = sin_sq_theta.item<float>() > 0.0 ? two_theta / sin_theta : 2.0 * torch::ones_like(sin_theta);
    // 3
    return k * torch::stack({q1, q2, q3}, 0);
  }

  // 3 x 3 -> 3
  inline at::Tensor RotationToAngleAxis(const at::Tensor rotation_matrix, float eps)
  {
    return QuaternionToAngleAxis(RotationToQuaternion(rotation_matrix, eps));
  }

  template <typename Scalar>
  inline void UpdateDepth(const at::Tensor dpt_map_bias, const at::Tensor dpt_jac_code,
                          const at::Tensor code, const Scalar &dpt_scale, at::Tensor &dpt_map)
  {
    torch::NoGradGuard no_grad;
    dpt_map = dpt_scale * (dpt_map_bias + torch::matmul(dpt_jac_code, code).reshape({1, 1, dpt_map_bias.size(2), dpt_map_bias.size(3)}));
    return;
  }

  template <typename Scalar>
  inline void RoughCorrectDepthScale(const at::Tensor video_mask, const typename Keyframe<Scalar>::Ptr &kf, const typename Keyframe<Scalar>::Ptr &reference_kf)
  {
    const at::Tensor ratio = torch::sum(torch::abs(reference_kf->dpt_map * video_mask)) / torch::sum(torch::abs(kf->dpt_map * video_mask));
    {
      std::unique_lock<std::shared_mutex> lock(kf->mutex);
      kf->dpt_scale = kf->dpt_scale * ratio.item<Scalar>();
      UpdateDepth(kf->dpt_map_bias, kf->dpt_jac_code, kf->code, kf->dpt_scale, kf->dpt_map);
    }
    return;
  }

  inline void ComputeSpatialGrad(const at::Tensor feat_map, at::Tensor &feat_map_grad)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    namespace F = torch::nn::functional;
    const long height = feat_map.size(2);
    const long width = feat_map.size(3);
    const at::Tensor padded_feat_map = F::pad(feat_map, F::PadFuncOptions({1, 1, 1, 1}).mode(torch::kReplicate));
    const at::Tensor feat_map_grad_x = 0.5 * (padded_feat_map.index({Slice(), Slice(), Slice(1, height + 1), Slice(2, width + 2)}) -
                                              padded_feat_map.index({Slice(), Slice(), Slice(1, height + 1), Slice(None, width)}));
    const at::Tensor feat_map_grad_y = 0.5 * (padded_feat_map.index({Slice(), Slice(), Slice(2, height + 2), Slice(1, width + 1)}) -
                                              padded_feat_map.index({Slice(), Slice(), Slice(None, height), Slice(1, width + 1)}));
    // TOOD: maybe we dont need to do the grad border correction, because we won't sample points near the border anyway.
    // 1 x 2*C_feat x H x W
    feat_map_grad = torch::cat({feat_map_grad_x, feat_map_grad_y}, 1);
    return;
  }

  inline void GenerateValidLocations(const at::Tensor valid_mask, const PinholeCamera<float> &camera,
                                     at::Tensor &sampled_normalized_locations_2d,
                                     at::Tensor &sampled_locations_1d, at::Tensor &sampled_locations_homo)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;

    // // N x 2
    // const at::Tensor hw_2d_locations = torch::nonzero(valid_mask.reshape({static_cast<long>(camera.height()),
    //                                                                        static_cast<long>(camera.width())}) > 0.9);
    // N
    // sampled_locations_1d = (hw_2d_locations.index({Slice(), 0}) * camera.width() + hw_2d_locations.index({Slice(), 1})).to(torch::kLong);

    sampled_locations_1d = torch::nonzero(valid_mask.reshape({-1}) > 0.5).reshape({-1});

    const at::Tensor sampled_locations_2d_x = torch::fmod(sampled_locations_1d, (float)camera.width());
    const at::Tensor sampled_locations_2d_y = torch::floor(sampled_locations_1d / (float)camera.width());

    // 1 x 1 x N x 2
    sampled_normalized_locations_2d = torch::stack({(sampled_locations_2d_x + 0.5) * (2.0 / camera.width()) - 1.0,
                                                    (sampled_locations_2d_y + 0.5) * (2.0 / camera.height()) - 1.0},
                                                   1)
                                          .reshape({1, 1, -1, 2});

    // N
    const at::Tensor sampled_locations_homo_x = (sampled_locations_2d_x - camera.u0()) / camera.fx();
    const at::Tensor sampled_locations_homo_y = (sampled_locations_2d_y - camera.v0()) / camera.fy();
    // N x 3
    sampled_locations_homo = torch::stack({sampled_locations_homo_x, sampled_locations_homo_y,
                                           torch::ones_like(sampled_locations_homo_x)},
                                          1);

    return;
  }

  inline at::Tensor SO3Hat(const at::Tensor omega)
  {
    at::Tensor omega_hat = torch::zeros({3, 3}, omega.options());
    omega_hat.index_put_({0, 1}, -omega.index({2}));
    omega_hat.index_put_({0, 2}, omega.index({1}));

    omega_hat.index_put_({1, 0}, omega.index({2}));
    omega_hat.index_put_({1, 2}, -omega.index({0}));

    omega_hat.index_put_({2, 0}, -omega.index({1}));
    omega_hat.index_put_({2, 1}, omega.index({0}));

    return omega_hat;
  }

  template <typename Scalar>
  Eigen::Matrix<Scalar, 3, 3> so3_hat(const Eigen::Matrix<Scalar, 3, 1> &omega)
  {
    Eigen::Matrix<Scalar, 3, 3> omega_hat;
    omega_hat << 0.0, -omega(2, 0), omega(1, 0),
        omega(2, 0), 0.0, -omega(0, 0),
        -omega(1, 0), omega(0, 0), 0.0;

    return omega_hat;
  }

  template <typename Scalar>
  void se3_exp(const Eigen::Matrix<Scalar, 3, 1> &omega, const Eigen::Matrix<Scalar, 3, 1> &v,
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

  inline void SE3Exp(const at::Tensor omega, const at::Tensor v, at::Tensor &R, at::Tensor &t)
  {
    const at::Tensor theta = omega.norm();
    const at::Tensor sintheta = theta.sin();
    const at::Tensor costheta = theta.cos();
    const at::Tensor normalized_omega = theta.item<float>() > 0 ? omega / theta : torch::rand_like(omega);

    const at::Tensor normalized_omega_hat = SO3Hat(normalized_omega);
    const at::Tensor normalized_omega_hat_sq = torch::matmul(normalized_omega_hat, normalized_omega_hat);

    const at::Tensor A = sintheta;
    const at::Tensor B = 1.0 - costheta;
    const at::Tensor C = theta - sintheta;

    const at::Tensor identity = torch::eye(3, omega.options());
    // 3 x 3
    R = identity + A * normalized_omega_hat + B * normalized_omega_hat_sq;
    const at::Tensor V = identity + (B / theta) * normalized_omega_hat + (C / theta) * normalized_omega_hat_sq;
    // 3 x 1
    t = torch::matmul(V, v.reshape({3, 1}));

    return;
  }

  template <typename Scalar>
  inline Scalar PoseDistance(const Sophus::SE3<Scalar> &pose_a, const Sophus::SE3<Scalar> &pose_b,
                             Scalar trs_wgt, Scalar rot_wgt)
  {
    Sophus::SE3<Scalar> relpose;
    RelativePose(pose_a, pose_b, relpose);
    // ignore the roll as a pure roll-rotated frame would not provide new information to SLAM system
    Scalar drot = relpose.so3().log().head(2).norm();
    Scalar dtrs = relpose.translation().norm();
    return dtrs * trs_wgt + drot * rot_wgt;
  }

  template <typename Scalar, int Row, int Col>
  inline at::Tensor SophusMatrixToTensor(Sophus::Matrix<Scalar, Row, Col> mat, const torch::TensorOptions &options)
  {
    c10::ScalarType dtype;
    if (std::is_same<float, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kFloat32;
    }
    else if (std::is_same<double, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kDouble;
    }
    else
    {
      assert(false);
    }

    // matrix is column-major by default
    return torch::from_blob(static_cast<Scalar *>(mat.data()), {mat.cols(), mat.rows()},
                            options.dtype(dtype).device(torch::kCPU))
        .to(options.device())
        .to(options.dtype())
        .permute({1, 0})
        .clone();
  }

  template <typename Scalar, int Row>
  inline at::Tensor SophusVectorToTensor(Sophus::Vector<Scalar, Row> vec, const torch::TensorOptions &options)
  {
    c10::ScalarType dtype;
    if (std::is_same<float, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kFloat32;
    }
    else if (std::is_same<double, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kDouble;
    }
    else
    {
      assert(false);
    }

    return torch::from_blob(static_cast<Scalar *>(vec.data()), {vec.size(), 1},
                            options.dtype(dtype).device(torch::kCPU))
        .to(options.device())
        .to(options.dtype())
        .clone();
  }

  template <typename Scalar, int Row, int Col>
  inline at::Tensor EigenMatToTensor(Eigen::Matrix<Scalar, Row, Col> matrix, const torch::TensorOptions &options)
  {
    c10::ScalarType dtype;
    if (std::is_same<float, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kFloat32;
    }
    else if (std::is_same<double, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kDouble;
    }
    else
    {
      assert(false);
    }

    return torch::from_blob(static_cast<Scalar *>(matrix.data()), {Col, Row},
                            options.device(torch::kCPU).dtype(dtype))
        .to(options.device())
        .to(options.dtype())
        .permute({1, 0})
        .clone();
  }

  template <typename Scalar, int Row>
  inline at::Tensor EigenVectorToTensor(Eigen::Matrix<Scalar, Row, 1> vector, const torch::TensorOptions &options)
  {
    c10::ScalarType dtype;
    if (std::is_same<float, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kFloat32;
    }
    else if (std::is_same<double, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kDouble;
    }
    else
    {
      assert(false);
    }

    return torch::from_blob(static_cast<Scalar *>(vector.data()), {vector.size()},
                            options.dtype(dtype).device(torch::kCPU))
        .to(options.device())
        .to(options.dtype())
        .clone();
  }

  template <typename Scalar, int Row, int Col, int Option>
  inline void TensorToEigenMatrix(const at::Tensor tensor, Eigen::Matrix<Scalar, Row, Col, Option> &eig_matrix)
  {
    // permute tensor to change the memory order
    // default eigen matrix is column-major

    c10::ScalarType dtype;
    if (std::is_same<float, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kFloat32;
    }
    else if (std::is_same<double, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kDouble;
    }
    else
    {
      assert(false);
    }

    std::memcpy(static_cast<Scalar *>(eig_matrix.data()),
                tensor.permute({1, 0}).to(torch::kCPU).to(dtype).contiguous().data_ptr(), sizeof(Scalar) * tensor.numel());
    return;
  }

  inline at::Tensor ComputeTransformedDepthJacPoseTR(const at::Tensor sampled_locations_3d)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    long num_points = sampled_locations_3d.size(0);
    const at::Tensor zeros = torch::zeros({num_points}, sampled_locations_3d.options());
    const at::Tensor ones = torch::ones({num_points}, sampled_locations_3d.options());
    // N x 1 x 6
    return torch::stack({zeros, zeros, ones, sampled_locations_3d.index({Slice(), 1}), -sampled_locations_3d.index({Slice(), 0}), zeros}, 1).reshape({num_points, 1, 6});
  }

  inline at::Tensor ComputeTransformedPointJacPoseTR(const at::Tensor sampled_locations_3d)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    long num_points = sampled_locations_3d.size(0);
    const at::Tensor zeros = torch::zeros({num_points}, sampled_locations_3d.options());
    const at::Tensor ones = torch::ones({num_points}, sampled_locations_3d.options());

    const at::Tensor row0 = torch::stack({ones, zeros, zeros, zeros, sampled_locations_3d.index({Slice(), 2}), -sampled_locations_3d.index({Slice(), 1})}, 1);
    const at::Tensor row1 = torch::stack({zeros, ones, zeros, -sampled_locations_3d.index({Slice(), 2}), zeros, sampled_locations_3d.index({Slice(), 0})}, 1);
    const at::Tensor row2 = torch::stack({zeros, zeros, ones, sampled_locations_3d.index({Slice(), 1}), -sampled_locations_3d.index({Slice(), 0}), zeros}, 1);
    // N x 3 x 6
    return torch::stack({row0, row1, row2}, 1);
  }

  inline at::Tensor ComputeTransformedDepthJacDepth(const at::Tensor sampled_rotated_locations_homo)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    long num_points = sampled_rotated_locations_homo.size(0);

    return sampled_rotated_locations_homo.index({Slice(), 2}).reshape({num_points, 1, 1});
  }

  inline at::Tensor ComputeTransformedPointJacDepth(const at::Tensor sampled_rotated_locations_homo)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    long num_points = sampled_rotated_locations_homo.size(0);

    return sampled_rotated_locations_homo.reshape({num_points, 3, 1});
  }

  template <typename Scalar>
  inline at::Tensor ComputeProj2DLocationsJacPoseTR(const at::Tensor sampled_locations_3d, const Scalar &fx, const Scalar &fy)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    long num_points = sampled_locations_3d.size(0);

    const at::Tensor zeros = torch::zeros({num_points}, sampled_locations_3d.options());
    const at::Tensor inv_z = 1.0 / sampled_locations_3d.index({Slice(), 2});
    const at::Tensor x_z = inv_z * sampled_locations_3d.index({Slice(), 0});
    const at::Tensor y_z = inv_z * sampled_locations_3d.index({Slice(), 1});

    const at::Tensor row_0 = torch::stack({fx * inv_z, zeros, -fx * x_z * inv_z, -fx * x_z * y_z, fx * (1.0 + torch::square(x_z)), -fx * y_z}, 1);
    const at::Tensor row_1 = torch::stack({zeros, fy * inv_z, -fy * y_z * inv_z, -fy * (1.0 + torch::square(y_z)), fy * x_z * y_z, fy * x_z}, 1);
    // N x 2 x 6
    return torch::stack({row_0, row_1}, 1);
  }

  template <typename Scalar>
  inline at::Tensor ComputeProj2DLocationsJacDepth(const at::Tensor sampled_rotated_locations_homo, const at::Tensor sampled_locations_3d,
                                                   const Scalar &fx, const Scalar &fy)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    long num_points = sampled_rotated_locations_homo.size(0);
    const at::Tensor inv_z = 1.0 / sampled_locations_3d.index({Slice(), 2});

    const at::Tensor item_0 = fx * (sampled_rotated_locations_homo.index({Slice(), 0}) * inv_z -
                                    sampled_locations_3d.index({Slice(), 0}) * sampled_rotated_locations_homo.index({Slice(), 2}) * torch::square(inv_z));

    const at::Tensor item_1 = fy * (sampled_rotated_locations_homo.index({Slice(), 1}) * inv_z -
                                    sampled_locations_3d.index({Slice(), 1}) * sampled_rotated_locations_homo.index({Slice(), 2}) * torch::square(inv_z));

    return torch::stack({item_0, item_1}, 1).reshape({num_points, 2, 1});
  }

  template <typename Scalar>
  inline void GenerateLocationsForJac(const at::Tensor sampled_dpts_0, const at::Tensor sampled_locations_homo_0, const at::Tensor rotation,
                                      const at::Tensor translation, const Scalar &eps, at::Tensor &sampled_rotated_locations_homo,
                                      at::Tensor &clamped_sampled_locations_3d_in_1, at::Tensor &sampled_locations_2d_in_1,
                                      at::Tensor &pos_depth_mask_1, const PinholeCamera<Scalar> &camera,
                                      bool normalize_locations_2d_output)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    // N x 3
    long num_points = sampled_locations_homo_0.size(0);
    // N x 3
    sampled_rotated_locations_homo = torch::matmul(rotation, sampled_locations_homo_0.permute({1, 0})).permute({1, 0});
    // N x 3
    const at::Tensor sampled_locations_3d_in_1 = sampled_dpts_0.reshape({-1, 1}) * sampled_rotated_locations_homo + translation.reshape({1, 3});

    Locations3DNegDepthClamp(sampled_locations_3d_in_1, eps, pos_depth_mask_1, clamped_sampled_locations_3d_in_1);

    // N x 3
    const at::Tensor sampled_locations_homo_in_1 = clamped_sampled_locations_3d_in_1 /
                                                   clamped_sampled_locations_3d_in_1.index({Slice(), Slice(2, 3)});
    // N x 1
    const at::Tensor sampled_locations_2d_x = sampled_locations_homo_in_1.index({Slice(), Slice(0, 1)}) * camera.fx() + camera.u0();
    const at::Tensor sampled_locations_2d_y = sampled_locations_homo_in_1.index({Slice(), Slice(1, 2)}) * camera.fy() + camera.v0();
    if (normalize_locations_2d_output)
    {
      // 1 x 1 x N x 2
      sampled_locations_2d_in_1 = torch::cat({(sampled_locations_2d_x + 0.5f) * (2.0f / camera.width()) - 1.0,
                                              (sampled_locations_2d_y + 0.5f) * (2.0f / camera.height()) - 1.0},
                                             1)
                                      .reshape({1, 1, num_points, 2});
    }
    else
    {
      // N x 2
      sampled_locations_2d_in_1 = torch::cat({sampled_locations_2d_x, sampled_locations_2d_y}, 1);
    }

    return;
  }

  template <typename Scalar>
  inline void GenerateLocationsForJacReproj(const at::Tensor sampled_dpts_0, const at::Tensor sampled_locations_homo_0, const at::Tensor rotation,
                                            const at::Tensor translation, const Scalar &eps, at::Tensor &sampled_rotated_locations_homo,
                                            at::Tensor &sampled_locations_3d_in_1, at::Tensor &sampled_locations_2d_in_1,
                                            at::Tensor &pos_depth_mask_1, const PinholeCamera<Scalar> &camera,
                                            bool normalize_locations_2d_output)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    // N x 3
    long num_points = sampled_locations_homo_0.size(0);
    // N x 3
    sampled_rotated_locations_homo = torch::matmul(rotation, sampled_locations_homo_0.permute({1, 0})).permute({1, 0});
    // N x 3
    sampled_locations_3d_in_1 = sampled_dpts_0.reshape({-1, 1}) * sampled_rotated_locations_homo + translation.reshape({1, 3});

    Locations3DSmallDepthClamp(sampled_locations_3d_in_1, eps, pos_depth_mask_1);

    // N x 3
    const at::Tensor sampled_locations_homo_in_1 = sampled_locations_3d_in_1 /
                                                   sampled_locations_3d_in_1.index({Slice(), Slice(2, 3)});
    // N x 1
    const at::Tensor sampled_locations_2d_x = sampled_locations_homo_in_1.index({Slice(), Slice(0, 1)}) * camera.fx() + camera.u0();
    const at::Tensor sampled_locations_2d_y = sampled_locations_homo_in_1.index({Slice(), Slice(1, 2)}) * camera.fy() + camera.v0();
    if (normalize_locations_2d_output)
    {
      // 1 x 1 x N x 2
      sampled_locations_2d_in_1 = torch::cat({(sampled_locations_2d_x + 0.5f) * (2.0f / camera.width()) - 1.0,
                                              (sampled_locations_2d_y + 0.5f) * (2.0f / camera.height()) - 1.0},
                                             1)
                                      .reshape({1, 1, num_points, 2});
    }
    else
    {
      // N x 2
      sampled_locations_2d_in_1 = torch::cat({sampled_locations_2d_x, sampled_locations_2d_y}, 1);
    }

    return;
  }

  template <typename Scalar>
  inline void GenerateReproj2DLocationsNoClamp(const at::Tensor sampled_dpts_0,
                                               const at::Tensor sampled_locations_homo_0, const at::Tensor rotation,
                                               const at::Tensor translation, const Scalar &eps,
                                               at::Tensor &sampled_locations_2d_in_1,
                                               at::Tensor &pos_depth_mask_1, const PinholeCamera<Scalar> &camera,
                                               const bool &normalize_locations_2d_output)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    // N x 3
    long num_points = sampled_locations_homo_0.size(0);
    const at::Tensor sampled_rotated_locations_homo = torch::matmul(rotation, sampled_locations_homo_0.permute({1, 0})).permute({1, 0});

    const at::Tensor sampled_locations_3d_in_1 = sampled_dpts_0.reshape({-1, 1}) * sampled_rotated_locations_homo + translation.reshape({1, 3});

    // N x 1
    pos_depth_mask_1 = sampled_locations_3d_in_1.index({Slice(), Slice(2, 3)}) > eps;

    // N x 3
    const at::Tensor sampled_locations_homo_in_1 = sampled_locations_3d_in_1 /
                                                   sampled_locations_3d_in_1.index({Slice(), Slice(2, 3)});
    // N x 1
    const at::Tensor sampled_locations_2d_x = sampled_locations_homo_in_1.index({Slice(), Slice(0, 1)}) * camera.fx() + camera.u0();
    const at::Tensor sampled_locations_2d_y = sampled_locations_homo_in_1.index({Slice(), Slice(1, 2)}) * camera.fy() + camera.v0();

    if (normalize_locations_2d_output)
    {
      // 1 x 1 x N x 2
      sampled_locations_2d_in_1 = torch::cat({(sampled_locations_2d_x + 0.5f) * (2.0f / camera.width()) - 1.0,
                                              (sampled_locations_2d_y + 0.5f) * (2.0f / camera.height()) - 1.0},
                                             1)
                                      .reshape({1, 1, num_points, 2});
    }
    else
    {
      // N x 2
      sampled_locations_2d_in_1 = torch::cat({sampled_locations_2d_x, sampled_locations_2d_y}, 1);
    }

    return;
  }

  template <typename Scalar>
  inline void SophusSE3ToTensor(const Sophus::SE3<Scalar> &pose, at::Tensor &rotation, at::Tensor &translation, const torch::TensorOptions &options)
  {
    torch::NoGradGuard no_grad;
    // auto pose_matrix = pose.so3();
    Eigen::Matrix<Scalar, 3, 3> rotation_matrix = pose.so3().matrix();
    Eigen::Matrix<Scalar, 3, 1> translation_vector = pose.translation();

    c10::ScalarType dtype;
    if (std::is_same<float, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kFloat32;
    }
    else if (std::is_same<double, typename std::remove_cv<Scalar>::type>::value)
    {
      dtype = torch::kDouble;
    }
    else
    {
      assert(false);
    }

    rotation = torch::from_blob(static_cast<Scalar *>(rotation_matrix.data()), {3, 3},
                                options.device(torch::kCPU).dtype(dtype))
                   .to(options.device())
                   .to(options.dtype())
                   .permute({1, 0})
                   .clone();
    translation = torch::from_blob(static_cast<Scalar *>(translation_vector.data()), {3, 1},
                                   options.device(torch::kCPU).dtype(dtype))
                      .to(options.device())
                      .to(options.dtype())
                      .clone();
    return;
  }

  template <typename Scalar>
  inline void SE3ImageWarping(const Sophus::SE3<Scalar> &pose_10, const PinholeCamera<Scalar> &camera,
                              const at::Tensor img_1, const at::Tensor dpt_map_0, const at::Tensor valid_mask,
                              at::Tensor &warped_img_1, at::Tensor &warped_valid_mask_1, const Scalar &dpt_eps)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    namespace F = torch::nn::functional;

    at::Tensor rotation, translation;
    SophusSE3ToTensor(pose_10, rotation, translation, img_1.options());

    const long channel = img_1.size(1);
    const long height = img_1.size(2);
    const long width = img_1.size(3);

    at::Tensor sampled_normalized_locations_2d_0, sampled_locations_1d_0, sampled_locations_homo_0;
    // N, 3 x N
    GenerateValidLocations(valid_mask, camera, sampled_normalized_locations_2d_0, sampled_locations_1d_0, sampled_locations_homo_0);
    // 1 x N
    const at::Tensor sampled_dpts_0 = dpt_map_0.reshape({height * width}).index({sampled_locations_1d_0}).reshape({1, -1});

    at::Tensor sampled_normalized_locations_2d_in_1, pos_depth_mask_in_1;
    // 1 x 1 x N x 2, N x 1
    GenerateReproj2DLocationsNoClamp(sampled_dpts_0, sampled_locations_homo_0, rotation, translation, dpt_eps,
                                     sampled_normalized_locations_2d_in_1, pos_depth_mask_in_1, camera, true);

    // C x N
    const at::Tensor sampled_img_1 = F::grid_sample(img_1, sampled_normalized_locations_2d_in_1,
                                                    F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false))
                                         .reshape({channel, -1});
    // 1 x N
    const at::Tensor sampled_valid_mask_1 = F::grid_sample(valid_mask, sampled_normalized_locations_2d_in_1,
                                                           F::GridSampleFuncOptions().mode(torch::kNearest).padding_mode(torch::kZeros).align_corners(false))
                                                .reshape({1, -1}) *
                                            pos_depth_mask_in_1.reshape({1, -1});

    warped_img_1 = torch::zeros_like(img_1).reshape({channel, height * width});
    warped_valid_mask_1 = torch::zeros_like(valid_mask).reshape({1, height * width});

    warped_img_1.index_put_({Slice(), sampled_locations_1d_0}, sampled_img_1);
    warped_valid_mask_1.index_put_({Slice(), sampled_locations_1d_0}, sampled_valid_mask_1);

    warped_img_1 = warped_img_1.reshape({1, channel, height, width});
    warped_valid_mask_1 = warped_valid_mask_1.reshape({1, 1, height, width});
  }

  template <typename Scalar>
  Scalar CorrectDepthScale(const typename df::Keyframe<Scalar>::Ptr &kf_to_scale,
                           const typename df::Keyframe<Scalar>::Ptr &reference_kf,
                           const Sophus::SE3<Scalar> &relpose_cur_ref,
                           const Scalar dpt_eps)
  {
    torch::NoGradGuard no_grad;
    using namespace torch::indexing;
    namespace F = torch::nn::functional;

    at::Tensor rotation, translation;
    SophusSE3ToTensor<Scalar>(relpose_cur_ref, rotation, translation, reference_kf->dpt_map_bias.options());

    auto camera = (*(reference_kf->camera_pyramid_ptr))[0];

    // 1 x 1 x H x W
    const at::Tensor valid_mask_1 = (*(kf_to_scale->video_mask_ptr)).reshape({1, 1, static_cast<long>(camera.height()), static_cast<long>(camera.width())});
    // N x 3
    const at::Tensor valid_locations_homo_0 = reference_kf->valid_locations_homo;
    // N
    const at::Tensor valid_locations_1d_0 = reference_kf->valid_locations_1d;

    at::Tensor valid_dpts_0;
    {
      std::shared_lock<std::shared_mutex> lock(reference_kf->mutex);
      // N
      valid_dpts_0 = reference_kf->dpt_map.reshape({-1}).index({valid_locations_1d_0});
    }

    // N x 3
    const at::Tensor valid_rotated_locations_homo = torch::matmul(rotation, valid_locations_homo_0.permute({1, 0})).permute({1, 0});
    // N x 3
    const at::Tensor valid_locations_3d_in_1 = valid_dpts_0.reshape({-1, 1}) * valid_rotated_locations_homo + translation.reshape({1, 3});
    at::Tensor pos_depth_mask_1;
    pos_depth_mask_1 = valid_locations_3d_in_1.index({Slice(), Slice(2, 3)}) > dpt_eps;

    // N x 3
    const at::Tensor valid_locations_homo_in_1 = valid_locations_3d_in_1 /
                                                 valid_locations_3d_in_1.index({Slice(), Slice(2, 3)});
    // N x 1
    const at::Tensor valid_locations_2d_x = valid_locations_homo_in_1.index({Slice(), Slice(0, 1)}) * camera.fx() + camera.u0();
    const at::Tensor valid_locations_2d_y = valid_locations_homo_in_1.index({Slice(), Slice(1, 2)}) * camera.fy() + camera.v0();
    // 1 x 1 x N x 2
    const at::Tensor valid_locations_2d_in_1 = torch::cat({(valid_locations_2d_x + 0.5f) * (2.0f / camera.width()) - 1.0,
                                                           (valid_locations_2d_y + 0.5f) * (2.0f / camera.height()) - 1.0},
                                                          1)
                                                   .reshape({1, 1, -1, 2});
    // N
    const at::Tensor valid_dpts_1 = F::grid_sample(kf_to_scale->dpt_map_bias.reshape({1, 1,
                                                                                      static_cast<long>(camera.height()), static_cast<long>(camera.width())}),
                                                   valid_locations_2d_in_1,
                                                   F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(false))
                                        .reshape({-1});
    const at::Tensor valid_valid_mask_1 = F::grid_sample(valid_mask_1, valid_locations_2d_in_1,
                                                         F::GridSampleFuncOptions().mode(torch::kNearest).padding_mode(torch::kZeros).align_corners(false))
                                              .reshape({-1}) *
                                          pos_depth_mask_1;

    at::Tensor ratios = ((valid_locations_3d_in_1.index({Slice(), 2}) * valid_valid_mask_1) / valid_dpts_1).reshape({-1});
    const at::Tensor nonzeros = torch::nonzero(valid_valid_mask_1 > 0.5).reshape({-1});
    const at::Tensor selected_ratios = ratios.index({nonzeros});
    const at::Tensor valid_selected_ratios = selected_ratios.index({torch::nonzero(torch::isnan(selected_ratios) < 0.5).reshape({-1})});
    at::Tensor ratio = torch::median(valid_selected_ratios);

    // if (std::isnan(ratio.item<Scalar>()))
    // {
    //   LOG(FATAL) << "[CorrectDepthScale] NaN found in depth ratio, nonzeros number: " << nonzeros.size(0) << " sum of nan valid depths : " << torch::sum(torch::isnan(valid_dpts_1)) << " num of invalid filtered ratios : " << torch::sum(torch::isnan(valid_selected_ratios)) << " num of ratios: " << selected_ratios.size(0) << " sum of valid depths: " << torch::sum(valid_dpts_1);
    // }
    return ratio.item<Scalar>();
  }

}

#endif