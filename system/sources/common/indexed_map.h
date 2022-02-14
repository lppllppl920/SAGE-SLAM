#ifndef DF_INDEXED_MAP_H_
#define DF_INDEXED_MAP_H_

namespace df
{

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <glog/logging.h>

template <typename Item, typename IdType>
class IndexedMap
{
public:
  typedef std::unordered_map<IdType, Item> ContainerT;
  typedef std::vector<IdType> IdList;

  /* Default constructor, does nothing */
  IndexedMap() : lastid_(0) {}

  /* Get an item by an id */
  virtual Item Get(const IdType& id)
  {
    if (!Exists(id))
      LOG(FATAL) << "[Get] Requesting item with id " << id << " which is not in the container";
    return map_[id];
  }

  virtual const Item Get(const IdType& id) const
  {
    if (!Exists(id))
      LOG(FATAL) << "[Get] Requesting item with id " << id << " which is not in the container";
    return map_.at(id);
  }

  /* Check whether an item with this id exists */
  virtual bool Exists(const IdType& id) const
  {
    return map_.find(id) != map_.end();
  }

  /*
   * Add an existing item to the map. Assign a new id to it.
   * Return the id.
   */
  virtual IdType Add(const Item& item)
  {
    map_[++lastid_] = item;
    ids_.push_back(lastid_);
    return lastid_;
  }

  /* Remove a value by Id */
  virtual void Remove(IdType id)
  {
    if (!Exists(id))
      LOG(FATAL) << "[Get] Attempting to remove non-existent item with id " << id;
    map_.erase(id);
    // The following one assumes ids are all unique
    auto pos = std::find(ids_.begin(), ids_.end(), id);
    if (pos != ids_.end()) {
    	ids_.erase(pos);
    } else{
    	LOG(FATAL) << "[Get] Attempting to remove non-existent item in id list with id " << id;
    }

  }

  /* Delete everything */
  virtual void Clear()
  {
    map_.clear();
    ids_.clear();
    lastid_ = 0;
  }

  virtual long Size() const { return map_.size(); }
  virtual const IdList& Ids() const { return ids_; }
  virtual const IdType LastId() const { return lastid_; }

protected:
  IdType lastid_;
  ContainerT map_;
  IdList ids_;
};

} // namespace df

#endif // DF_INDEXED_MAP_H_
