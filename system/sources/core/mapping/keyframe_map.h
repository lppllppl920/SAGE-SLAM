#ifndef DF_KEYFRAME_MAP_H_
#define DF_KEYFRAME_MAP_H_

#include <map>
#include <vector>
#include <memory>
#include <torch/torch.h>

#include "keyframe.h"
#include "indexed_map.h"

namespace df
{

  template <typename FrameT, typename IdType = typename FrameT::IdType>
  class FrameGraph : public IndexedMap<FrameT, IdType>
  {
  public:
    typedef IndexedMap<FrameT, IdType> Base;
    typedef std::pair<IdType, IdType> LinkT;
    typedef std::vector<LinkT> LinkContainer;

    void AddLink(IdType first, IdType second)
    {
      // emplace_back avoids creating a temporary object, which is better than push_back in this case
      links_.emplace_back(first, second);
    }

    /* Remove a value by Id */
    void Remove(IdType id) override
    {
      Base::Remove(id);

      // remove all links related to this frame
      for (uint i = 0; i < links_.size(); ++i)
      {
        if (links_[i].first == id || links_[i].second == id)
        {
          links_.erase(links_.begin() + i);
        }
      }
    }

    void Clear() override
    {
      Base::Clear();
      links_.clear();
    }

    LinkContainer &GetLinks() { return links_; }

    FrameT &Last() { return this->map_[this->LastId()]; }
    // get the ids of the frames that are connected to the specified one
    std::vector<IdType> GetConnections(IdType id, bool directed = false)
    {
      std::vector<IdType> conns;
      for (auto &c : links_)
      {
        if (c.first == id)
        {
          conns.push_back(c.second);
        }

        if (c.second == id && !directed)
        {
          conns.push_back(c.first);
        }
      }
      return conns;
    }

    bool LinkExists(IdType id0, IdType id1)
    {
      for (auto &c : links_)
      {
        if ((c.first == id0 && c.second == id1) ||
            (c.first == id1 && c.second == id0))
        {
          return true;
        }
      }
      return false;
    }

    typename Base::ContainerT::iterator begin() { return this->map_.begin(); }
    typename Base::ContainerT::iterator end() { return this->map_.end(); }

  private:
    LinkContainer links_;
  };

  template <typename Scalar>
  class Map
  {
  public:
    typedef Map<Scalar> This;
    typedef std::shared_ptr<This> Ptr;
    typedef Frame<Scalar> FrameT;
    typedef Keyframe<Scalar> KeyframeT;
    typedef typename FrameT::IdType FrameId;
    typedef typename KeyframeT::Ptr KeyframePtr;
    typedef typename FrameT::Ptr FramePtr;
    typedef FrameGraph<FramePtr, FrameId> FrameGraphT;
    typedef FrameGraph<KeyframePtr, FrameId> KeyframeGraphT;

    void Clear()
    {
      frames.Clear();
      keyframes.Clear();
    }

    void AddFrame(FramePtr fr) { fr->id = frames.Add(fr); }
    void AddKeyframe(KeyframePtr kf) { kf->id = keyframes.Add(kf); }

    long NumKeyframes() const { return keyframes.Size(); }
    long NumFrames() const { return frames.Size(); }

    FrameGraph<FramePtr, FrameId> frames;
    FrameGraph<KeyframePtr, FrameId> keyframes;
  };

} // namespace df

#endif // DF_KEYFRAME_MAP_H_
