#include "work.h"

namespace df
{
  namespace work
  {

    int Work::next_id = 0;

    Work::~Work() {}

    Work::Work()
    {
      id = "[" + std::to_string(next_id++) + "]";
    }

    Work::Ptr Work::AddChild(Ptr child)
    {
      child_ = child;
      return child;
    }

    Work::Ptr Work::RemoveChild()
    {
      auto child = child_;
      child_.reset();
      return child;
    }

  } // namespace work
} // namespace df
