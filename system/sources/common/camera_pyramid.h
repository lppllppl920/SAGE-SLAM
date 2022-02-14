#ifndef DF_CAMERA_PYRAMID_H_
#define DF_CAMERA_PYRAMID_H_

#include <vector>
#include "pinhole_camera.h"

namespace df
{

template <typename T>
class CameraPyramid
{
  typedef df::PinholeCamera<T> CameraT;
public:
  
  CameraPyramid() {}

  CameraPyramid(const CameraT& cam, std::size_t levels) : levels_(levels)
  {
    for (std::size_t i = 0; i < levels; ++i)
    {
      // better than push_back in the sense that no potential redundant object copy is required.
      cameras_.emplace_back(cam);

      if (i != 0)
      {
        std::size_t new_width = cameras_[i-1].width() / 2;
        std::size_t new_height = cameras_[i-1].height() / 2;
        cameras_[i].ResizeViewport(new_width, new_height);
      }
    }
  }

  inline const CameraT& operator[](int i) const
  {
    return cameras_[i];
  }

  inline CameraT& operator[](int i)
  {
    return cameras_[i];
  }

  std::size_t Levels() const { return levels_; }

private:
  std::vector<CameraT> cameras_;
  std::size_t levels_;
};

} // namespace df

template <typename T>
std::ostream& operator<<(std::ostream& os, const df::CameraPyramid<T>& pyr)
{
  os << "CameraPyramid:" << std::endl;
  for (std::size_t i = 0; i < pyr.Levels(); ++i)
    os << "Level " << i << " " << pyr[i] << std::endl;
  return os;
}

#endif // DF_CAMERA_PYRAMID_H_
