#ifndef DF_CAMERA_INTERFACE_H_
#define DF_CAMERA_INTERFACE_H_

#include <opencv2/opencv.hpp>

#include "pinhole_camera.h"

// forward declarations
namespace cv { class Mat; }

/**
 * Custom exception to detect whether a certain
 * feature is not supported by your camera
 */
class UnsupportedFeatureException : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

namespace df
{
namespace drivers
{

/**
  * Interface class for all cameras
  * that will work with this system
  */
class CameraInterface
{
public:
  CameraInterface() {}
  virtual ~CameraInterface() {}

  virtual void SetGain(float gain) { throw UnsupportedFeatureException("Gain not supported by this camera"); }
  virtual void SetGainAuto(bool on) { throw UnsupportedFeatureException("Gain not supported by this camera"); }
  virtual float GetGain() { throw UnsupportedFeatureException("Gain not supported by this camera"); }
  virtual void SetShutter(float shutter) { throw UnsupportedFeatureException("Shutter not supported by this camera"); }
  virtual void SetShutterAuto(bool on) { throw UnsupportedFeatureException("Shutter not supported by this camera"); }
  virtual float GetShutter() { throw UnsupportedFeatureException("Shutter not supported by this camera"); }
  virtual cv::Mat GetMask() { throw UnsupportedFeatureException("GetMask not supported by this camera"); }
  virtual bool SupportsDepth() { return false; }
  virtual bool HasIntrinsics() { return false; }
  virtual bool HasMore() = 0;

  virtual df::PinholeCamera<float> GetIntrinsics() { return df::PinholeCamera<float>{}; }

  /**
   * Grabs new frames from the camera and preprocesses
   * them to a standard form
   *
   * @param timestamp timestamp of the rgb frame
   * @param img Output 3-channel color image (uint8)
   * @param dpt Optional output depth map, one channel float
   */
  virtual void GrabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr) = 0;
};

} // namespace drivers
} // namespace df

#endif // DF_CAMERA_INTERFACE_H_

