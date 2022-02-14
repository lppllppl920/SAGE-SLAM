#ifndef DF_WORK_MANAGER_H_
#define DF_WORK_MANAGER_H_

#include <list>
#include <functional>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <unordered_map>

#include "work.h"
// #include "df_work.h"

namespace df
{
  namespace work
  {

    class WorkManager
    {
    public:
      typedef Work::Ptr WorkPtr;

      template <typename T, class... Args>
      WorkPtr AddWork(Args &&...args)
      {
        auto work = std::make_shared<T>(std::forward<Args>(args)...);
        return AddWork(work);
      }

      WorkPtr AddWork(WorkPtr work);

      void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                       gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                       gtsam::Values &var_init,
                       gtsam::Values &var_update);

      void DistributeIndices(gtsam::FastVector<gtsam::FactorIndex> indices);
      void Remove(std::function<bool(WorkPtr)> f);
      void Erase(std::function<bool(WorkPtr)> f);

      void Update();
      void SignalNoRelinearize();

      void PrintWork();
      bool Empty() const { return work_.empty(); }
      void Clear();

    private:
      std::list<WorkPtr> work_;
      // map of work id to pointer of work
      std::unordered_map<std::string, WorkPtr> work_map_;
      // map of work id to the size of the work (number of factors)
      std::unordered_map<std::string, int> last_new_factors_;
    };

  } // namespace work
} // namespace df

#endif // DF_WORK_MANAGER_H_
