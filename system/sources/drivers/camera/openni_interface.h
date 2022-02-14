#ifndef DF_OPENNI_INTERFACE_H_
#define DF_OPENNI_INTERFACE_H_

#include <memory>
#include <glog/logging.h>
#include <OpenNIDriver.hpp>
#include <opencv2/opencv.hpp>

#include "live_interface.h"
#include "camera_interface_factory.h"

namespace drivers { namespace camera { class OpenNI; } }

namespace df
{
namespace drivers
{

class OpenNIInterface : public LiveInterface
{
public:
  OpenNIInterface(std::size_t camera_id);
  virtual ~OpenNIInterface();

  virtual void SetGain(float gain) override;
  virtual void SetGainAuto(bool on) override;
  virtual float GetGain() override;
  virtual void SetShutter(float shutter) override;
  virtual void SetShutterAuto(bool on) override;
  virtual float GetShutter() override;

  virtual void GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr) override;

  virtual bool SupportsDepth() override { return true; }
  virtual bool HasIntrinsics() override { return true; }

  virtual df::PinholeCamera<float> GetIntrinsics() override;

private:
  std::unique_ptr<::drivers::camera::OpenNI> driver;
};

class OpenNIInterfaceFactory : public SpecificInterfaceFactory
{
public:
  virtual std::unique_ptr<CameraInterface> FromUrlParams(const std::string& url_params) override;
  virtual std::string GetUrlPattern(const std::string& prefix_tag) override;
  virtual std::string GetPrefix() override;

private:
  const std::string url_prefix = "openni";
  const std::string url_params = "[cameraId]";
};

} // namespace drivers
} // namespace df

#endif // DF_OPENNI_INTERFACE_H_
