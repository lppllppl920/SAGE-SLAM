#include "df_work.h"

namespace df
{
  namespace work
  {

    /** XT
   * The error should beequal to the log of gaussian likelihood, e.g. \f$ 0.5(h(x)-z)^2/sigma^2 \f$ (?)
   */
    template <typename Scalar, int CS>
    InitVariables<Scalar, CS>::InitVariables(KeyframePtr kf)
        : first_(true), fr_(kf)
    {
      // Here it assumes that the initial code vector is zero, which is true in our case
      gtsam::Vector zero_code = gtsam::Vector::Zero(CS);
      var_init_.insert(PoseKey(kf->id), kf->pose_wk);
      var_init_.insert(CodeKey(kf->id), zero_code);
      var_init_.insert(ScaleKey(kf->id), kf->dpt_scale);
      name_ = kf->Name();
    }

    template <typename Scalar, int CS>
    InitVariables<Scalar, CS>::InitVariables(KeyframePtr kf, Scalar pose_prior)
        : InitVariables(kf)
    {
      // Additional pose prior added besides those defined in the constructor above
      typedef gtsam::noiseModel::Diagonal DiagonalNoise;
      typedef gtsam::PriorFactor<Sophus::SE3f> PriorFactor;

      gtsam::Vector prior_sigmas = gtsam::Vector::Constant(6, 1, pose_prior);
      auto prior_noise = DiagonalNoise::Sigmas(prior_sigmas);
      priors_.emplace_shared<PriorFactor>(PoseKey(kf->id), Sophus::SE3f{}, prior_noise);
    }

    // Frame object only has camera pose variable
    template <typename Scalar, int CS>
    InitVariables<Scalar, CS>::InitVariables(FramePtr fr)
        : first_(true), fr_(fr)
    {
      gtsam::Key pose_key = fr_->IsKeyframe() ? PoseKey(fr_->id) : AuxPoseKey(fr_->id);
      var_init_.insert(pose_key, fr_->pose_wk);
      name_ = fr_->Name();
    }

    template <typename Scalar, int CS>
    InitVariables<Scalar, CS>::~InitVariables() {}

    template <typename Scalar, int CS>
    void InitVariables<Scalar, CS>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                                gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                                gtsam::Values &var_init,
                                                gtsam::Values &var_update)
    {
      if (first_)
      {
        new_factors += priors_;
        var_init.insert(var_init_);
        first_ = false;
      }
      
      if (fr_->reinitialize_count.load(std::memory_order_relaxed) >= 1)
      {
        // If some variables are manually reinitialized, we need to update the variables inside ISAM2 correspondingly
        gtsam::Key pose_key = fr_->IsKeyframe() ? PoseKey(fr_->id) : AuxPoseKey(fr_->id);
        // Update both the pose and scale
        var_update.insert(pose_key, fr_->pose_wk);
        var_update.insert(ScaleKey(fr_->id), fr_->dpt_scale);
      }
    }

    template <typename Scalar, int CS>
    void InitVariables<Scalar, CS>::Update()
    {
    }

    template <typename Scalar, int CS>
    std::string InitVariables<Scalar, CS>::Name()
    {
      return Id() + " InitVariables " + name_;
    }

    // explicit instantiation
    template class InitVariables<float, DF_CODE_SIZE>;

    /* ************************************************************************* */
    template <typename Scalar>
    OptimizeWork<Scalar>::OptimizeWork(int iters, bool remove_after)
        : first_(true),
          iters_(iters),
          active_level_(0),
          remove_after_(remove_after) {}
    template <typename Scalar>
    void OptimizeWork<Scalar>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                           gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                           gtsam::Values &var_init,
                                           gtsam::Values &var_update)
    {
      if (active_level_ <= -1 && remove_after_)
      {
        remove_indices.insert(remove_indices.begin(), last_indices_.begin(),
                              last_indices_.end());
      }

      if (first_)
      {
        first_ = false;
        new_factors += ConstructFactors();
        remove_indices.insert(remove_indices.begin(), last_indices_.begin(),
                              last_indices_.end());
      }
    }

    template <typename Scalar>
    gtsam::NonlinearFactorGraph OptimizeWork<Scalar>::ConstructFactors()
    {
      return gtsam::NonlinearFactorGraph();
    }

    template <typename Scalar>
    void OptimizeWork<Scalar>::Update()
    {
      // update the active level based on specified number of iterations to finish this factor optimization
      if (active_level_ >= 0 && --iters_ < 0)
      {
        active_level_ -= 1;
      }
    }

    template <typename Scalar>
    bool OptimizeWork<Scalar>::Finished() const
    {
      // this indicator is used to remove the work from the work manager and work list,
      // will this also remove the work from the GTSAM graph?
      return active_level_ <= -1 && remove_after_;
    }

    template <typename Scalar>
    void OptimizeWork<Scalar>::SignalNoRelinearize()
    {
      if (!first_)
      {
        active_level_ -= 1;
      }
    }

    template <typename Scalar>
    void OptimizeWork<Scalar>::SignalRemove()
    {
      active_level_ = -1;
    }

    template <typename Scalar>
    void OptimizeWork<Scalar>::LastFactorIndices(gtsam::FastVector<gtsam::FactorIndex> &indices)
    {
      last_indices_ = indices;
    }

    template class OptimizeWork<float>;

    /* ************************************************************************* */
    template <typename Scalar, int CS>
    OptimizePhoto<Scalar, CS>::OptimizePhoto(const KeyframePtr &kf, const FramePtr &fr, const int &iters,
                                             const CamPyramid &cam_pyr, const std::vector<Scalar> &factor_weights,
                                             const Scalar &dpt_eps, const bool &remove_after, const bool display_stats)
        : OptimizeWork<Scalar>(iters, remove_after),
          kf_(kf), fr_(fr),
          cam_pyr_(cam_pyr), factor_weights_(factor_weights),
          dpt_eps_(dpt_eps), display_stats_(display_stats) {}

    template <typename Scalar, int CS>
    OptimizePhoto<Scalar, CS>::~OptimizePhoto() {}

    template <typename Scalar, int CS>
    void OptimizePhoto<Scalar, CS>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                                gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                                gtsam::Values &var_init,
                                                gtsam::Values &var_update)
    {

      OptimizeWork<Scalar>::Bookkeeping(new_factors, remove_indices, var_init, var_update);
    }

    template <typename Scalar, int CS>
    gtsam::NonlinearFactorGraph OptimizePhoto<Scalar, CS>::ConstructFactors()
    {
      DCHECK_GE(this->active_level_, 0);
      gtsam::NonlinearFactorGraph graph;

      gtsam::Key pose0_key = PoseKey(kf_->id);
      gtsam::Key code0_key = CodeKey(kf_->id);
      gtsam::Key scale0_key = ScaleKey(kf_->id);
      gtsam::Key pose1_key = fr_->IsKeyframe() ? PoseKey(fr_->id) : AuxPoseKey(fr_->id);

      graph.emplace_shared<PhotoFactor>(cam_pyr_, kf_, fr_,
                                        pose0_key, pose1_key, code0_key, scale0_key,
                                        factor_weights_, dpt_eps_, display_stats_);
      return graph;
    }

    template <typename Scalar, int CS>
    std::string OptimizePhoto<Scalar, CS>::Name()
    {
      std::stringstream ss;
      ss << this->Id() << " OptimizePhoto " << kf_->Name() << " -> " << fr_->Name()
         << " iters = " << this->iters_ << " finished = " << this->Finished();
      return ss.str();
    }

    template <typename Scalar, int CS>
    bool OptimizePhoto<Scalar, CS>::Involves(OptimizePhoto::FramePtr ptr) const
    {
      return fr_ == ptr || kf_ == ptr;
    }

    // explicit instantiation
    // this seems to avoid having to copy function codes each time an object with the speicified type here is defined.
    // https://stackoverflow.com/questions/2351148/explicit-template-instantiation-when-is-it-used

    // DF_CODE_SIZE is defined in CMakeLists. Change the value of it to fit our network architecture
    template class OptimizePhoto<float, DF_CODE_SIZE>;

    /* ************************************************************************* */
    template <typename Scalar, int CS>
    OptimizeGeo<Scalar, CS>::OptimizeGeo(const KeyframePtr &kf0,
                                         const KeyframePtr &kf1,
                                         int iters,
                                         const df::PinholeCamera<Scalar> &cam,
                                         const Scalar &factor_weight,
                                         const Scalar &loss_param,
                                         const Scalar &dpt_eps,
                                         const bool &remove_after)
        : OptimizeWork<Scalar>(iters, remove_after),
          kf0_(kf0), kf1_(kf1),
          cam_(cam), factor_weight_(factor_weight),
          loss_param_(loss_param), dpt_eps_(dpt_eps)
    {
    }

    template <typename Scalar, int CS>
    OptimizeGeo<Scalar, CS>::~OptimizeGeo() {}

    template <typename Scalar, int CS>
    void OptimizeGeo<Scalar, CS>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                              gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                              gtsam::Values &var_init,
                                              gtsam::Values &var_update)
    {
      OptimizeWork<Scalar>::Bookkeeping(new_factors, remove_indices, var_init, var_update);
    }

    template <typename Scalar, int CS>
    gtsam::NonlinearFactorGraph OptimizeGeo<Scalar, CS>::ConstructFactors()
    {
      gtsam::NonlinearFactorGraph graph;
      auto pose0_key = PoseKey(kf0_->id);
      auto pose1_key = PoseKey(kf1_->id);
      auto code0_key = CodeKey(kf0_->id);
      auto code1_key = CodeKey(kf1_->id);
      auto scale0_key = ScaleKey(kf0_->id);
      auto scale1_key = ScaleKey(kf1_->id);

      graph.emplace_shared<GeoFactor>(cam_, kf0_, kf1_, pose0_key, pose1_key,
                                      code0_key, code1_key, scale0_key, scale1_key,
                                      factor_weight_, loss_param_, dpt_eps_);
      return graph;
    }

    template <typename Scalar, int CS>
    bool OptimizeGeo<Scalar, CS>::Involves(FramePtr ptr) const
    {
      return kf0_ == ptr || kf1_ == ptr;
    }

    template <typename Scalar, int CS>
    std::string OptimizeGeo<Scalar, CS>::Name()
    {
      std::stringstream ss;
      ss << this->Id() << " OptimizeGeo " << kf0_->Name() << " -> " << kf1_->Name()
         << " iters = " << this->iters_ << " finished = " << this->Finished();
      return ss.str();
    }

    // explicit instantiation
    template class OptimizeGeo<float, DF_CODE_SIZE>;

    /* *************************************************** */
    template <typename Scalar, int CS>
    OptimizeRep<Scalar, CS>::OptimizeRep(const KeyframePtr &kf, const FramePtr &fr, const int &iters,
                                         const df::PinholeCamera<Scalar> &cam, const long &num_keypoints,
                                         const Scalar &cyc_consis_thresh, const Scalar &loss_param,
                                         const Scalar &factor_weight,
                                         const Scalar &dpt_eps,
                                         const bool &remove_after,
                                         const teaser::RobustRegistrationSolver::Params &teaser_params,
                                         const bool display_stats)
        : OptimizeWork<Scalar>(iters, remove_after),
          kf_(kf), fr_(fr),
          cam_(cam), num_keypoints_(num_keypoints),
          cyc_consis_thresh_(cyc_consis_thresh),
          loss_param_(loss_param),
          factor_weight_(factor_weight),
          dpt_eps_(dpt_eps), teaser_params_(teaser_params),
          display_stats_(display_stats), finished_(false) {}

    template <typename Scalar, int CS>
    OptimizeRep<Scalar, CS>::~OptimizeRep() {}

    template <typename Scalar, int CS>
    gtsam::NonlinearFactorGraph OptimizeRep<Scalar, CS>::ConstructFactors()
    {
      gtsam::NonlinearFactorGraph graph;
      gtsam::Key pose0_key = PoseKey(kf_->id);
      gtsam::Key code0_key = CodeKey(kf_->id);
      gtsam::Key pose1_key = fr_->IsKeyframe() ? PoseKey(fr_->id) : AuxPoseKey(fr_->id);
      gtsam::Key scale0_key = ScaleKey(kf_->id);

      boost::shared_ptr<RepFactor> factor = boost::make_shared<RepFactor>(cam_, kf_, fr_, pose0_key, pose1_key,
                                                                          code0_key, scale0_key, num_keypoints_,
                                                                          cyc_consis_thresh_, factor_weight_, loss_param_,
                                                                          dpt_eps_, teaser_params_, display_stats_);
      if (factor->InlierMatches() == 0)
      {
        VLOG(1) << "MATCHES ARE EMPTY, NOT ADDING FACTOR!";
        this->finished_ = true;
      }
      else
      {
        graph.add(factor);
      }

      return graph;
    }

    template <typename Scalar, int CS>
    bool OptimizeRep<Scalar, CS>::Finished() const
    {
      // XT finished_ will be true if the number of inlier matches is zero (then the factor will never get called I think)
      return OptimizeWork<Scalar>::Finished() || finished_;
    }

    template <typename Scalar, int CS>
    bool OptimizeRep<Scalar, CS>::Involves(FramePtr ptr) const
    {
      return fr_ == ptr || kf_ == ptr;
    }

    template <typename Scalar, int CS>
    std::string OptimizeRep<Scalar, CS>::Name()
    {
      std::stringstream ss;
      ss << this->Id() << " OptimizeRep " << kf_->Name() << " -> " << fr_->Name()
         << " iters = " << this->iters_ << " finished = " << this->Finished();
      return ss.str();
    }

    // explicit instantiation
    template class OptimizeRep<float, DF_CODE_SIZE>;

    /* *************************************************** */
    template <typename Scalar, int CS>
    OptimizeMatchGeo<Scalar, CS>::OptimizeMatchGeo(const KeyframePtr &kf0, const KeyframePtr &kf1, const int &iters,
                                                   const df::PinholeCamera<Scalar> &cam,
                                                   const long &num_keypoints,
                                                   const Scalar &cyc_consis_thresh, const Scalar &loss_param,
                                                   const Scalar &factor_weight,
                                                   const Scalar &dpt_eps,
                                                   const std::string robust_loss_type,
                                                   const bool &remove_after,
                                                   const teaser::RobustRegistrationSolver::Params &teaser_params,
                                                   const bool display_stats)
        : OptimizeWork<Scalar>(iters, remove_after),
          kf0_(kf0), kf1_(kf1),
          cam_(cam), num_keypoints_(num_keypoints),
          cyc_consis_thresh_(cyc_consis_thresh),
          loss_param_(loss_param),
          factor_weight_(factor_weight),
          dpt_eps_(dpt_eps), robust_loss_type_(robust_loss_type),
          teaser_params_(teaser_params),
          display_stats_(display_stats), finished_(false) {}

    template <typename Scalar, int CS>
    OptimizeMatchGeo<Scalar, CS>::~OptimizeMatchGeo() {}

    template <typename Scalar, int CS>
    void OptimizeMatchGeo<Scalar, CS>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                                   gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                                   gtsam::Values &var_init,
                                                   gtsam::Values &var_update)
    {
      OptimizeWork<Scalar>::Bookkeeping(new_factors, remove_indices, var_init, var_update);
    }

    template <typename Scalar, int CS>
    gtsam::NonlinearFactorGraph OptimizeMatchGeo<Scalar, CS>::ConstructFactors()
    {
      gtsam::NonlinearFactorGraph graph;
      gtsam::Key pose0_key = PoseKey(kf0_->id);
      gtsam::Key pose1_key = PoseKey(kf1_->id);

      gtsam::Key code0_key = CodeKey(kf0_->id);
      gtsam::Key code1_key = CodeKey(kf1_->id);

      gtsam::Key scale0_key = ScaleKey(kf0_->id);
      gtsam::Key scale1_key = ScaleKey(kf1_->id);

      boost::shared_ptr<MatchGeoFactor> factor =
          boost::make_shared<MatchGeoFactor>(cam_, kf0_, kf1_, pose0_key, pose1_key,
                                             code0_key, code1_key, scale0_key, scale1_key, num_keypoints_,
                                             cyc_consis_thresh_, factor_weight_, loss_param_,
                                             dpt_eps_, robust_loss_type_, teaser_params_,
                                             display_stats_);
      // I think this is a way to walk around how factors are constructed
      if (factor->InlierMatches() == 0)
      {
        VLOG(2) << "[OptimizeMatchGeo<Scalar, CS>::ConstructFactors] Matches are empty, not adding factor!";
        this->finished_ = true;
      }
      else
      {
        graph.add(factor);
      }

      return graph;
    }

    template <typename Scalar, int CS>
    bool OptimizeMatchGeo<Scalar, CS>::Finished() const
    {
      // finished_ will be true if the number of inlier matches is small (then the factor will never get called I think)
      return OptimizeWork<Scalar>::Finished() || finished_;
    }

    template <typename Scalar, int CS>
    bool OptimizeMatchGeo<Scalar, CS>::Involves(FramePtr ptr) const
    {
      return kf0_ == ptr || kf1_ == ptr;
    }

    template <typename Scalar, int CS>
    std::string OptimizeMatchGeo<Scalar, CS>::Name()
    {
      std::stringstream ss;
      ss << this->Id() << " OptimizeMatchGeo " << kf0_->Name() << " -> " << kf1_->Name()
         << " iters = " << this->iters_ << " finished = " << this->Finished();
      return ss.str();
    }

    // explicit instantiation
    template class OptimizeMatchGeo<float, DF_CODE_SIZE>;

    template <typename Scalar>
    OptimizeLoopMG<Scalar>::OptimizeLoopMG(
        const KeyframePtr &kf0, const KeyframePtr &kf1, const int &iters,
        const at::Tensor matched_unscaled_dpts_0,
        const at::Tensor matched_unscaled_dpts_1,
        const at::Tensor matched_locations_homo_0,
        const at::Tensor matched_locations_homo_1,
        const Scalar &loss_param, const Scalar &factor_weight,
        const Scalar &dpt_eps, const bool &remove_after)
        : OptimizeWork<Scalar>(iters, remove_after),
          kf0_(kf0), kf1_(kf1), loss_param_(loss_param),
          factor_weight_(factor_weight), dpt_eps_(dpt_eps),
          matched_unscaled_dpts_0_(matched_unscaled_dpts_0),
          matched_unscaled_dpts_1_(matched_unscaled_dpts_1),
          matched_locations_homo_0_(matched_locations_homo_0),
          matched_locations_homo_1_(matched_locations_homo_1)
    {
    }
    template <typename Scalar>
    OptimizeLoopMG<Scalar>::~OptimizeLoopMG() {}

    template <typename Scalar>
    void OptimizeLoopMG<Scalar>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                             gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                             gtsam::Values &var_init,
                                             gtsam::Values &var_update)
    {
      OptimizeWork<Scalar>::Bookkeeping(new_factors, remove_indices, var_init, var_update);
    }

    template <typename Scalar>
    gtsam::NonlinearFactorGraph OptimizeLoopMG<Scalar>::ConstructFactors()
    {
      gtsam::NonlinearFactorGraph graph;
      gtsam::Key pose0_key = PoseKey(kf0_->id);
      gtsam::Key pose1_key = PoseKey(kf1_->id);

      gtsam::Key scale0_key = ScaleKey(kf0_->id);
      gtsam::Key scale1_key = ScaleKey(kf1_->id);

      graph.emplace_shared<LoopMGFactorT>(kf0_, kf1_, pose0_key, pose1_key,
                                          scale0_key, scale1_key,
                                          matched_unscaled_dpts_0_,
                                          matched_unscaled_dpts_1_,
                                          matched_locations_homo_0_,
                                          matched_locations_homo_1_,
                                          factor_weight_, loss_param_,
                                          dpt_eps_);
      return graph;
    }

    template <typename Scalar>
    bool OptimizeLoopMG<Scalar>::Involves(FramePtr ptr) const
    {
      return kf0_ == ptr || kf1_ == ptr;
    }

    template <typename Scalar>
    std::string OptimizeLoopMG<Scalar>::Name()
    {
      std::stringstream ss;
      ss << this->Id() << " OptimizeLoopMG " << kf0_->Name() << " -> " << kf1_->Name()
         << " iters = " << this->iters_ << " finished = " << this->Finished();
      return ss.str();
    }

    // explicit instantiation
    template class OptimizeLoopMG<float>;

    /* *************************************************** */
    template <typename Scalar>
    OptimizeScale<Scalar>::OptimizeScale(const KeyframePtr &kf, const int &iters,
                                         const Scalar &init_scale, const Scalar &factor_weight,
                                         const bool &remove_after)
        : OptimizeWork<Scalar>(iters, remove_after),
          kf_(kf), init_scale_(init_scale), factor_weight_(factor_weight)
    {
    }

    template <typename Scalar>
    OptimizeScale<Scalar>::~OptimizeScale() {}

    template <typename Scalar>
    void OptimizeScale<Scalar>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                            gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                            gtsam::Values &var_init,
                                            gtsam::Values &var_update)
    {
      OptimizeWork<Scalar>::Bookkeeping(new_factors, remove_indices, var_init, var_update);
    }

    template <typename Scalar>
    gtsam::NonlinearFactorGraph OptimizeScale<Scalar>::ConstructFactors()
    {
      gtsam::NonlinearFactorGraph graph;
      gtsam::Key scale_key = ScaleKey(kf_->id);

      graph.emplace_shared<ScaleFactorT>(kf_, scale_key, init_scale_, factor_weight_);
      return graph;
    }

    template <typename Scalar>
    bool OptimizeScale<Scalar>::Involves(FramePtr ptr) const
    {
      return kf_ == ptr;
    }

    template <typename Scalar>
    std::string OptimizeScale<Scalar>::Name()
    {
      std::stringstream ss;
      ss << this->Id() << " OptimizeScale " << kf_->Name() << " finished = " << this->Finished();
      return ss.str();
    }

    // explicit instantiation
    template class OptimizeScale<float>;

    /* *************************************************** */
    template <typename Scalar>
    OptimizePose<Scalar>::OptimizePose(const KeyframePtr &kf,
                                       const int &iters,
                                       const PoseT &target_pose,
                                       const Scalar &factor_weight,
                                       const bool &remove_after)
        : OptimizeWork<Scalar>(iters, remove_after),
          kf_(kf), target_pose_(target_pose), factor_weight_(factor_weight)
    {
    }

    template <typename Scalar>
    OptimizePose<Scalar>::~OptimizePose() {}

    template <typename Scalar>
    void OptimizePose<Scalar>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                           gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                           gtsam::Values &var_init,
                                           gtsam::Values &var_update)
    {
      OptimizeWork<Scalar>::Bookkeeping(new_factors, remove_indices, var_init, var_update);
    }

    template <typename Scalar>
    gtsam::NonlinearFactorGraph OptimizePose<Scalar>::ConstructFactors()
    {
      gtsam::NonlinearFactorGraph graph;
      gtsam::Key pose_key = PoseKey(kf_->id);

      graph.emplace_shared<PoseFactorT>(kf_, pose_key, target_pose_, factor_weight_);
      return graph;
    }

    template <typename Scalar>
    bool OptimizePose<Scalar>::Involves(FramePtr ptr) const
    {
      return kf_ == ptr;
    }

    template <typename Scalar>
    std::string OptimizePose<Scalar>::Name()
    {
      std::stringstream ss;
      ss << this->Id() << " OptimizePose " << kf_->Name() << " finished = " << this->Finished();
      return ss.str();
    }

    // explicit instantiation
    template class OptimizePose<float>;

    /* *************************************************** */
    template <typename Scalar, int CS>
    OptimizeCode<Scalar, CS>::OptimizeCode(const KeyframePtr &kf,
                                           const int &iters, const CodeT &init_code,
                                           const Scalar &factor_weight, const bool &remove_after)
        : OptimizeWork<Scalar>(iters, remove_after),
          kf_(kf), init_code_(init_code), factor_weight_(factor_weight)
    {
    }

    template <typename Scalar, int CS>
    OptimizeCode<Scalar, CS>::~OptimizeCode() {}

    template <typename Scalar, int CS>
    void OptimizeCode<Scalar, CS>::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                               gtsam::Values &var_init,
                                               gtsam::Values &var_update)
    {
      OptimizeWork<Scalar>::Bookkeeping(new_factors, remove_indices, var_init, var_update);
    }

    template <typename Scalar, int CS>
    gtsam::NonlinearFactorGraph OptimizeCode<Scalar, CS>::ConstructFactors()
    {
      gtsam::NonlinearFactorGraph graph;
      gtsam::Key code_key = CodeKey(kf_->id);

      graph.emplace_shared<CodeFactorT>(kf_, code_key, init_code_, factor_weight_);
      return graph;
    }

    template <typename Scalar, int CS>
    bool OptimizeCode<Scalar, CS>::Involves(FramePtr ptr) const
    {
      return kf_ == ptr;
    }

    template <typename Scalar, int CS>
    std::string OptimizeCode<Scalar, CS>::Name()
    {
      std::stringstream ss;
      ss << this->Id() << " OptimizeCode " << kf_->Name() << " finished = " << this->Finished();
      return ss.str();
    }

    // explicit instantiation
    template class OptimizeCode<float, DF_CODE_SIZE>;

  } // namespace work
} // namespace df
