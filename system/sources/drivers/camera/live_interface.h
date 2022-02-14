#ifndef DF_LIVE_INTERFACE_H_
#define DF_LIVE_INTERFACE_H_

#include "camera_interface.h"
#include "pinhole_camera.h"

// forward declarations
namespace cv { class Mat; }

namespace df
{
namespace drivers
{

/**
  * Interface class for all cameras
  * that will work with this system
  */
class LiveInterface : public CameraInterface
{
public:
  LiveInterface() {}
  virtual ~LiveInterface() {}

  virtual bool HasMore() override { return true; }

  virtual void SetGain(float gain) override { throw UnsupportedFeatureException("Gain not supported by this camera"); }
  virtual void SetGainAuto(bool on) override { throw UnsupportedFeatureException("Gain not supported by this camera"); }
  virtual float GetGain() override { throw UnsupportedFeatureException("Gain not supported by this camera"); }
  virtual void SetShutter(float shutter) override { throw UnsupportedFeatureException("Shutter not supported by this camera"); }
  virtual void SetShutterAuto(bool on) override { throw UnsupportedFeatureException("Shutter not supported by this camera"); }
  virtual float GetShutter() override { throw UnsupportedFeatureException("Shutter not supported by this camera"); }
};

} // namespace drivers
} // namespace df

#endif // DF_LIVE_INTERFACE_H_

