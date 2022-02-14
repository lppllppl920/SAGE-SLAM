#ifndef DF_KEYFRAME_H_
#define DF_KEYFRAME_H_

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <shared_mutex>
#include <atomic>

#include "frame.h"

namespace df
{

  template <typename Scalar>
  class Keyframe : public Frame<Scalar>
  {
  public:
    typedef Keyframe<Scalar> This;
    typedef Frame<Scalar> Base;
    typedef std::shared_ptr<Keyframe<Scalar>> Ptr;
    typedef typename Keyframe<Scalar>::IdType KeyframeId;

    Keyframe()
        : Base(), local_loop_searched(false), global_loop_searched(false)
    {
    }

    Keyframe(const Keyframe &other)
        : Base(other),
          temporal_connections(other.temporal_connections),
          local_loop_connections(other.local_loop_connections),
          global_loop_connections(other.global_loop_connections)
    {
      local_loop_searched = other.local_loop_searched.load(std::memory_order_relaxed);
      global_loop_searched = other.global_loop_searched.load(std::memory_order_relaxed);
    }

    virtual ~Keyframe() {}
    virtual typename Base::Ptr Clone() override
    {
      return std::make_shared<This>(*this);
    }

    virtual std::string Name() override { return "kf" + std::to_string(this->id); }

    virtual bool IsKeyframe() override { return true; }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<KeyframeId> temporal_connections;

    std::vector<KeyframeId> local_loop_connections;
    std::atomic<bool> local_loop_searched;

    std::vector<KeyframeId> global_loop_connections;
    std::atomic<bool> global_loop_searched;
  };

} // namespace df

#endif // DF_KEYFRAME_H_
