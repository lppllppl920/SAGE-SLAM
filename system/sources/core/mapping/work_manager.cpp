#include "work_manager.h"

namespace df
{
  namespace work
  {

    WorkManager::WorkPtr WorkManager::AddWork(WorkPtr work)
    {
      work_.push_back(work);
      work_map_.insert({work->Id(), work});
      return work;
    }

    void WorkManager::Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                                  gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                                  gtsam::Values &var_init,
                                  gtsam::Values &var_update)
    {
      // factors added this iteration
      // will be used in consuming the factor indices after ISAM2::update step
      last_new_factors_.clear();
      // collect bookkeeping from all work
      for (auto &work : work_)
      {
        gtsam::NonlinearFactorGraph add_factors;
        gtsam::FastVector<gtsam::FactorIndex> rem_factors;
        gtsam::Values init;
        gtsam::Values update;
        work->Bookkeeping(add_factors, rem_factors, init, update);

        // keep track which work items added factors
        if (!add_factors.empty())
        {
          last_new_factors_.insert({work->Id(), add_factors.size()});
        }

        // add new factors, vars, etc
        new_factors += add_factors;
        remove_indices.insert(remove_indices.end(), rem_factors.begin(), rem_factors.end());
        var_init.insert(init);
        var_update.insert(update);
      }
    }
    // this is where the factor indices created from GTSAM graph are distributed to works
    void WorkManager::DistributeIndices(gtsam::FastVector<gtsam::FactorIndex> indices)
    {
      for (auto &kv : last_new_factors_)
      {
        std::string id = kv.first;
        auto work = work_map_[id];
        int n = kv.second;

        // first N goes to id
        gtsam::FastVector<gtsam::FactorIndex> ind(indices.begin(), indices.begin() + n);
        if (work)
        {
          work->LastFactorIndices(ind);
        }

        // remove
        indices.erase(indices.begin(), indices.begin() + n);
      }
      last_new_factors_.clear();
    }

    void WorkManager::Remove(std::function<bool(WorkPtr)> f)
    {
      auto it = std::remove_if(work_.begin(), work_.end(), f);
      for (auto ii = it; ii != work_.end(); ++ii)
        (*ii)->SignalRemove();
    }

    void WorkManager::Erase(std::function<bool(WorkPtr)> f)
    {
      // only the ones that do not fulfill f would be left intact and others will be removed
      auto it = std::remove_if(work_.begin(), work_.end(), f);
      work_.erase(it, work_.end());
    }

    void WorkManager::Update()
    {
      VLOG(3) << "[WorkManager::Update] Current work:";
      for (auto &work : work_)
        VLOG(3) << "[WorkManager::Update] " << work->Name();

      auto it = work_.begin();
      while (it != work_.end())
      {
        WorkPtr work = *it;

        // // in case we need to remove this work
        auto this_it = it;
        it++;
        work->Update();

        if (work->Finished())
        {
          VLOG(2) << "Work " << work->Name() << " has finished";
          work_map_.erase(work->Id());
          work_.erase(this_it);
        }
      }
    }

    void WorkManager::SignalNoRelinearize()
    {
      for (auto &work : work_)
        work->SignalNoRelinearize();
    }

    void WorkManager::PrintWork()
    {
      for (auto &work : work_)
        LOG(INFO) << "[WorkManager::PrintWork] " << work->Name();
    }

    void WorkManager::Clear()
    {
      work_.clear();
      work_map_.clear();
      last_new_factors_.clear();
    }

  } // namespace work
} // namespace df
