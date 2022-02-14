#ifndef DF_SCANNET_INETRFACE_H_
#define DF_SCANNET_INETRFACE_H_

#include "dataset_interface.h"
#include "camera_interface_factory.h"

namespace df
{
namespace drivers
{

class ScanNetInterface : public DatasetInterface
{
public:
  ScanNetInterface(std::string seq_path);
  virtual ~ScanNetInterface();

  virtual void GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr) override;
  virtual std::vector<DatasetFrame> GetAll() override;

  virtual bool SupportsDepth() override { return has_depth_; }
  virtual bool HasIntrinsics() override { return true; }
  virtual bool HasPoses() override { return true; }
  virtual bool HasMore() override { return current_frame_ < (int)img_files_.size()-1; }
  virtual std::vector<Sophus::SE3f> GetPoses() override { return poses_; }

  virtual df::PinholeCamera<float> GetIntrinsics() override;

private:
  void LoadFrame(int i, double& timestamp, cv::Mat* img, cv::Mat* dpt, Sophus::SE3f& pose);
  df::PinholeCamera<float> LoadIntrinsics(std::string seq_dir, int width, int height);
  std::vector<Sophus::SE3f> LoadPoses();
  std::size_t ProbeNumberOfFrames(std::string dir);

  std::string seq_path_;
  std::vector<std::string> img_files_;
  std::vector<std::string> dpt_files_;
  std::vector<std::string> pose_files_;

  std::vector<Sophus::SE3f> poses_;
  df::PinholeCamera<float> cam_;
  int current_frame_;
  bool has_depth_;
};

class ScanNetInterfaceFactory : public SpecificInterfaceFactory
{
public:
  virtual std::unique_ptr<CameraInterface> FromUrlParams(const std::string& url_params) override;
  virtual std::string GetUrlPattern(const std::string& prefix_tag) override;
  virtual std::string GetPrefix() override;

private:
  const std::string url_prefix = "scannet";
  const std::string url_params = "sequence_path";
};

} // namespace drivers
} // namespace df

#endif // DF_SCANNET_INETRFACE_H_
