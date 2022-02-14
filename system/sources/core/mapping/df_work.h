#ifndef DF_DF_WORK_H_
#define DF_DF_WORK_H_

#include <torch/torch.h>
#include <gtsam/slam/PriorFactor.h>
#include <teaser/registration.h>

#include "gtsam_utils.h"
#include "gtsam_traits.h"
#include "work.h"
#include "keyframe.h"
#include "camera_pyramid.h"
#include "photometric_factor.h"
#include "geometric_factor.h"
#include "match_geometry_factor.h"
#include "loop_mg_factor.h"
#include "scale_factor.h"
#include "pose_factor.h"
#include "code_factor.h"
#include "reprojection_factor.h"

namespace df
{
  template <typename Scalar, int CS>
  class PhotometricFactor;

  template <typename Scalar, int CS>
  class ReprojectionFactor;

  template <typename Scalar, int CS>
  class MatchGeometryFactor;

  template <typename Scalar>
  class LoopMGFactor;

  template <typename Scalar, int CS>
  class GeometricFactor;

  template <typename Scalar, int CS>
  class CodeFactor;

  template <typename Scalar>
  class ScaleFactor;

  template <typename Scalar>
  class PoseFactor;

  namespace work
  {

    class CallbackWork : public Work
    {
    public:
      typedef std::function<void()> CallbackT;
      CallbackWork(CallbackT f) : finished_(false), f_(f) {}

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override {}

      virtual void Update() override
      {
        f_();
        finished_ = true;
      }

      virtual bool Finished() const override
      {
        return finished_;
      }

      virtual std::string Name() override { return "CallbackWork"; }

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

      bool finished_;
      CallbackT f_;
    };

    template <typename Scalar, int CS>
    class InitVariables : public Work
    {
    public:
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef typename df::Frame<Scalar>::Ptr FramePtr;

      // constructor from keyframe
      InitVariables(KeyframePtr kf); // , Scalar code_prior

      // constructor from keyframe with zero-pose prior
      InitVariables(KeyframePtr kf, Scalar pose_prior); // Scalar code_prior,

      // constructor from frame
      InitVariables(FramePtr fr);

      virtual ~InitVariables();

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;

      virtual void Update() override;

      virtual bool Finished() const override
      {
        return false;
      }

      virtual std::string Name() override;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
      bool first_;
      FramePtr fr_;
      gtsam::Values var_init_;
      gtsam::NonlinearFactorGraph priors_;
      std::string name_;
    };

    // adds support for multi-scale optimization
    template <typename Scalar>
    class OptimizeWork : public Work
    {
    public:
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef std::vector<int> IterList;

      // constructor for single scale optimization
      OptimizeWork(int iters, bool remove_after = false);

      virtual ~OptimizeWork() {}

      // basic version of this func adds some factors
      // on first run and then keeps track of iterations
      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;

      // override this function to create simple
      // non pyramid levels
      virtual gtsam::NonlinearFactorGraph ConstructFactors();

      // counts iterations and descends pyramid levels
      virtual void Update() override;

      virtual bool Finished() const override;

      virtual void SignalNoRelinearize() override;
      virtual void SignalRemove() override;
      virtual void LastFactorIndices(gtsam::FastVector<gtsam::FactorIndex> &indices) override;

      virtual bool Involves(FramePtr ptr) const = 0;

      // bool IsCoarsestLevel() const;
      // bool IsNewLevelStart() const;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    protected:
      const gtsam::FastVector<gtsam::FactorIndex> &GetLastIndices() { return last_indices_; }

    protected:
      bool first_;
      int iters_;
      int active_level_;
      gtsam::FastVector<gtsam::FactorIndex> last_indices_;
      bool remove_after_;
    };

    // handles photo error
    template <typename Scalar, int CS>
    class OptimizePhoto : public OptimizeWork<Scalar>
    {
    public:
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef typename df::CameraPyramid<Scalar> CamPyramid;
      typedef df::PhotometricFactor<Scalar, CS> PhotoFactor;
      typedef typename OptimizeWork<Scalar>::IterList IterList;

      // constructor with a vector of iterations
      OptimizePhoto(const KeyframePtr &kf, const FramePtr &fr, const int &iters,
                    const CamPyramid &cam_pyr, const std::vector<Scalar> &factor_weights,
                    const Scalar &dpt_eps, const bool &remove_after, const bool display_stats = false);

      virtual ~OptimizePhoto();

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;

      gtsam::NonlinearFactorGraph ConstructFactors() override;
      virtual bool Involves(FramePtr ptr) const override;
      virtual std::string Name() override;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      KeyframePtr kf_;
      FramePtr fr_;
      df::CameraPyramid<Scalar> cam_pyr_;
      std::vector<Scalar> factor_weights_;
      Scalar dpt_eps_;

      bool display_stats_;
    };

    /*!
 * \brief Class that manages factors for geometric error
 */
    template <typename Scalar, int CS>
    class OptimizeGeo : public OptimizeWork<Scalar>
    {
    public:
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef df::GeometricFactor<Scalar, CS> GeoFactor;

      // constructor with single number of iterations
      OptimizeGeo(const KeyframePtr &kf0, const KeyframePtr &kf1,
                  int iters, const df::PinholeCamera<Scalar> &cam,
                  const Scalar &factor_weight, const Scalar &loss_param,
                  const Scalar &dpt_eps, const bool &remove_after);

      virtual ~OptimizeGeo();

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;

      virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
      virtual bool Involves(FramePtr ptr) const override;
      virtual std::string Name() override;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      KeyframePtr kf0_;
      KeyframePtr kf1_;
      df::PinholeCamera<Scalar> cam_;
      Scalar factor_weight_;
      Scalar loss_param_;
      Scalar dpt_eps_;
    };

    template <typename Scalar, int CS>
    class OptimizeRep : public OptimizeWork<Scalar>
    {
    public:
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef df::ReprojectionFactor<Scalar, CS> RepFactor;

      // constructor with single number of iterations
      OptimizeRep(const KeyframePtr &kf, const FramePtr &fr, const int &iters,
                  const df::PinholeCamera<Scalar> &cam,
                  const long &num_keypoints,
                  const Scalar &cyc_consis_thresh, const Scalar &loss_param,
                  const Scalar &factor_weight,
                  const Scalar &dpt_eps, const bool &remove_after,
                  const teaser::RobustRegistrationSolver::Params &teaser_params,
                  const bool display_stats = false);

      virtual ~OptimizeRep();

      virtual bool Finished() const override;
      virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
      virtual bool Involves(FramePtr ptr) const override;
      virtual std::string Name() override;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      KeyframePtr kf_;
      FramePtr fr_;
      df::PinholeCamera<Scalar> cam_;
      long num_keypoints_;
      Scalar cyc_consis_thresh_;
      Scalar loss_param_;
      Scalar factor_weight_;
      Scalar dpt_eps_;

      teaser::RobustRegistrationSolver::Params teaser_params_;

      bool display_stats_;
      bool finished_;
    };

    template <typename Scalar, int CS>
    class OptimizeMatchGeo : public OptimizeWork<Scalar>
    {
    public:
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef df::MatchGeometryFactor<Scalar, CS> MatchGeoFactor;

      // constructor with single number of iterations
      OptimizeMatchGeo(const KeyframePtr &kf0, const KeyframePtr &kf1, const int &iters,
                       const df::PinholeCamera<Scalar> &cam,
                       const long &num_keypoints,
                       const Scalar &cyc_consis_thresh, const Scalar &loss_param,
                       const Scalar &factor_weight,
                       const Scalar &dpt_eps, const std::string robust_loss_type,
                       const bool &remove_after,
                       const teaser::RobustRegistrationSolver::Params &teaser_params,
                       const bool display_stats = false);

      virtual ~OptimizeMatchGeo();

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;
      virtual bool Finished() const override;
      virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
      virtual bool Involves(FramePtr ptr) const override;
      virtual std::string Name() override;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      KeyframePtr kf0_, kf1_;
      df::PinholeCamera<Scalar> cam_;
      long num_keypoints_;
      Scalar cyc_consis_thresh_;
      Scalar loss_param_;
      Scalar factor_weight_;
      Scalar dpt_eps_;

      std::string robust_loss_type_;

      teaser::RobustRegistrationSolver::Params teaser_params_;

      bool display_stats_;
      bool finished_;
    };

    template <typename Scalar>
    class OptimizeScale : public OptimizeWork<Scalar>
    {
    public:
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef typename df::ScaleFactor<Scalar> ScaleFactorT;

      // constructor with single number of iterations
      OptimizeScale(const KeyframePtr &kf,
                    const int &iters, const Scalar &init_scale,
                    const Scalar &factor_weight, const bool &remove_after);

      virtual ~OptimizeScale();

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;
      virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
      virtual bool Involves(FramePtr ptr) const override;
      virtual std::string Name() override;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      KeyframePtr kf_;
      Scalar init_scale_;
      Scalar factor_weight_;
    };

    template <typename Scalar>
    class OptimizePose : public OptimizeWork<Scalar>
    {
    public:
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef Sophus::SE3<Scalar> PoseT;
      typedef typename df::PoseFactor<Scalar> PoseFactorT;

      // constructor with single number of iterations
      OptimizePose(const KeyframePtr &kf,
                   const int &iters,
                   const PoseT &target_pose,
                   const Scalar &factor_weight,
                   const bool &remove_after);

      virtual ~OptimizePose();

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;

      virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
      virtual bool Involves(FramePtr ptr) const override;
      virtual std::string Name() override;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      KeyframePtr kf_;
      PoseT target_pose_;
      Scalar factor_weight_;
    };

    template <typename Scalar, int CS>
    class OptimizeCode : public OptimizeWork<Scalar>
    {
    public:
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef typename df::CodeFactor<Scalar, CS> CodeFactorT;
      typedef gtsam::Vector CodeT;

      // constructor with single number of iterations
      OptimizeCode(const KeyframePtr &kf,
                   const int &iters, const CodeT &init_code,
                   const Scalar &factor_weight, const bool &remove_after);

      virtual ~OptimizeCode();

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;
      virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
      virtual bool Involves(FramePtr ptr) const override;
      virtual std::string Name() override;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      KeyframePtr kf_;
      CodeT init_code_;
      Scalar factor_weight_;
    };

    template <typename Scalar>
    class OptimizeLoopMG : public OptimizeWork<Scalar>
    {
    public:
      typedef typename df::Frame<Scalar>::Ptr FramePtr;
      typedef typename df::Keyframe<Scalar>::Ptr KeyframePtr;
      typedef df::LoopMGFactor<Scalar> LoopMGFactorT;

      // constructor with single number of iterations
      OptimizeLoopMG(const KeyframePtr &kf0, const KeyframePtr &kf1, const int &iters,
                     const at::Tensor matched_unscaled_dpts_0,
                     const at::Tensor matched_unscaled_dpts_1,
                     const at::Tensor matched_locations_homo_0,
                     const at::Tensor matched_locations_homo_1,
                     const Scalar &loss_param, const Scalar &factor_weight,
                     const Scalar &dpt_eps, const bool &remove_after);

      virtual ~OptimizeLoopMG();

      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) override;
      virtual gtsam::NonlinearFactorGraph ConstructFactors() override;
      virtual bool Involves(FramePtr ptr) const override;
      virtual std::string Name() override;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    private:
      KeyframePtr kf0_, kf1_;
      Scalar loss_param_;
      Scalar factor_weight_;
      Scalar dpt_eps_;
      at::Tensor matched_unscaled_dpts_0_;
      at::Tensor matched_unscaled_dpts_1_;
      at::Tensor matched_locations_homo_0_;
      at::Tensor matched_locations_homo_1_;
    };

  } // namespace work
} // namespace df

#endif // DF_DF_WORK_H_
