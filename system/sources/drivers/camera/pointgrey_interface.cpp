#include "pointgrey_interface.h"

namespace df
{
  namespace drivers
  {

    static InterfaceRegistrar<PointGreyInterfaceFactory> automatic;

    PointGreyInterface::PointGreyInterface(uint camera_id)
    {
      driver = std::make_unique<::drivers::camera::PointGrey>();

      try
      {
        driver->open(camera_id);
        driver->setCustomMode(::drivers::camera::EPixelFormat::PIXEL_FORMAT_RGB8, 0, 0, 0, 0, 2);

        const int fps = 30;

        // set frame rate
        driver->setFeaturePower(::drivers::camera::EFeature::FRAME_RATE, true);
        driver->setFeatureAuto(::drivers::camera::EFeature::FRAME_RATE, false);
        driver->setFeatureValueAbs(::drivers::camera::EFeature::FRAME_RATE, fps);

        // set gain
        driver->setFeaturePower(::drivers::camera::EFeature::GAIN, true);
        driver->setFeatureAuto(::drivers::camera::EFeature::GAIN, false);
        driver->setFeatureValueAbs(::drivers::camera::EFeature::GAIN, 0);

        // set shutter
        driver->setFeaturePower(::drivers::camera::EFeature::SHUTTER, true);
        driver->setFeatureAuto(::drivers::camera::EFeature::SHUTTER, false);
        driver->setFeatureValueAbs(::drivers::camera::EFeature::SHUTTER, 1000.f / fps);

        driver->start();
      }
      catch (std::exception &e)
      {
        throw std::runtime_error("Failed to open and configure PointGrey camera: " + std::string(e.what()));
      }
    }

    PointGreyInterface::~PointGreyInterface()
    {
      driver->stop();
      driver->close();
    }

    void PointGreyInterface::SetGain(float gain)
    {
      driver->setFeatureValueAbs(::drivers::camera::EFeature::GAIN, gain);
    }

    void PointGreyInterface::SetGainAuto(bool on)
    {
      driver->setFeatureAuto(::drivers::camera::EFeature::GAIN, on);
    }

    float PointGreyInterface::GetGain()
    {
      return driver->getFeatureValueAbs(::drivers::camera::EFeature::GAIN);
    }

    void PointGreyInterface::SetShutter(float shutter)
    {
      driver->setFeatureValueAbs(::drivers::camera::EFeature::SHUTTER, shutter);
    }

    void PointGreyInterface::SetShutterAuto(bool on)
    {
      driver->setFeatureAuto(::drivers::camera::EFeature::SHUTTER, on);
    }

    float PointGreyInterface::GetShutter()
    {
      return driver->getFeatureValueAbs(::drivers::camera::EFeature::SHUTTER);
    }

    void PointGreyInterface::GrabFrames(double &timestamp, cv::Mat *img, cv::Mat *dpt)
    {
      ::drivers::camera::FrameBuffer video_frame;

      if (!driver->captureFrame(video_frame))
      {
        throw std::runtime_error("Error grabbing frame");
      }

      if (img != nullptr)
      {
        cv::Mat ptr_mat(video_frame.getHeight(), video_frame.getWidth(), CV_8UC3, video_frame.getData());
        ptr_mat.copyTo(*img);
        cv::cvtColor(*img, *img, cv::COLOR_RGB2BGR);

        timestamp = static_cast<double>(video_frame.getTimeStamp());
      }

      if (dpt != nullptr)
      {
        throw UnsupportedFeatureException("Depth not supported by this camera");
      }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Factory class for this interface
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::unique_ptr<CameraInterface> PointGreyInterfaceFactory::FromUrlParams(const std::string &url_params)
    {
      // parse cameraid from params
      std::size_t camera_id = 0;
      if (url_params.length())
      {
        camera_id = std::stoul(url_params);
      }

      return std::make_unique<PointGreyInterface>(camera_id);
    }

    std::string PointGreyInterfaceFactory::GetUrlPattern(const std::string &prefix_tag)
    {
      return url_prefix + prefix_tag + url_params;
    }

    std::string PointGreyInterfaceFactory::GetPrefix()
    {
      return url_prefix;
    }

  } // namespace drivers
} // namespace df
