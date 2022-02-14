#ifndef DF_FILE_INTERFACE_H_
#define DF_FILE_INTERFACE_H_

#include <memory>
#include  <vector>
#include "dataset_interface.h"
#include "camera_interface_factory.h"

namespace drivers { namespace camera { class PointGrey; } }

namespace df
{
namespace drivers
{

class FileInterface : public DatasetInterface
{
public:
  FileInterface(const std::string& glob_path);
  virtual ~FileInterface();

  virtual void GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr) override;
  virtual bool HasMore() override;
  virtual bool HasPoses() override { return false; }
  virtual bool HasIntrinsics() override { return true; }

  virtual df::PinholeCamera<float> GetIntrinsics() override { return cam_; }
  virtual std::vector<DatasetFrame> GetAll() override { return std::vector<DatasetFrame>{}; } // unimplemented

private:
  int count_;
  std::vector<std::string> files_;
  df::PinholeCamera<float> cam_;
};

class FileInterfaceFactory : public SpecificInterfaceFactory
{
public:
  virtual std::unique_ptr<CameraInterface> FromUrlParams(const std::string& url_params) override;
  virtual std::string GetUrlPattern(const std::string& prefix_tag) override;
  virtual std::string GetPrefix() override;

private:
  const std::string url_prefix = "filesrc";
  const std::string url_params = "path_with_glob";
};


} // namespace drivers
} // namespace df

#endif // DF_FILE_INTERFACE_H_
