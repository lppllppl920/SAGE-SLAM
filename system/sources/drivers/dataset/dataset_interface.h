#ifndef DF_DATASET_INTERFACE_H_
#define DF_DATASET_INTERFACE_H_

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "camera_interface.h"

namespace df
{
namespace drivers
{

struct DatasetFrame
{
  double timestamp;
  cv::Mat img;
  cv::Mat dpt;
  Sophus::SE3f pose_wf;
};

class DatasetInterface : public CameraInterface
{
public:
  DatasetInterface() {}
  virtual ~DatasetInterface() {}
  
  virtual std::vector<DatasetFrame> GetAll() = 0;
  virtual std::vector<Sophus::SE3f> GetPoses() { return std::vector<Sophus::SE3f>{}; }
  virtual bool HasPoses() = 0;
  virtual bool HasMore() = 0;
};

} // namespace drivers
} // namespace df

#endif // DF_DATASET_INTERFACE_H_

