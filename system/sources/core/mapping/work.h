#ifndef DF_WORK_H_
#define DF_WORK_H_

#include <memory>
#include <string>
#include <torch/torch.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/base/FastVector.h>

// typedef FactorIndices so that we don't have to include ISAM2 header here
namespace gtsam
{
  typedef gtsam::FastVector<size_t> FactorIndices;
}

namespace df
{
  namespace work
  {

    class Work
    {
    public:
      typedef std::shared_ptr<Work> Ptr;

      Work();

      virtual ~Work();

      // interface that needs to be implemented
      // by child functions
      virtual void Bookkeeping(gtsam::NonlinearFactorGraph &new_factors,
                               gtsam::FastVector<gtsam::FactorIndex> &remove_indices,
                               gtsam::Values &var_init,
                               gtsam::Values &var_update) = 0;
      virtual void Update() = 0;
      virtual bool Finished() const = 0;
      virtual std::string Name() = 0;

      virtual void SignalNoRelinearize() {}
      virtual void SignalRemove() {}
      virtual void LastFactorIndices(gtsam::FastVector<gtsam::FactorIndex> &indices) {}

      template <class T, class... Args>
      Ptr AddChild(Args &&...args)
      {
        auto child = std::make_shared<T>(std::forward<Args>(args)...);
        return AddChild(child);
      }

      Ptr AddChild(Ptr child);
      Ptr RemoveChild();

      std::string Id() const { return id; }

    private:
      Ptr child_;
      std::string id;

      static int next_id;
    };

  } // namespace work
} // namespace df

#endif // DF_WORK_H_
