#ifndef DF_HDF5_INTERFACE_H_
#define DF_HDF5_INTERFACE_H_

#include <memory>
#include <vector>
#include <memory>
#include <fstream>
#include <glob.h>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <H5Cpp.h>

#include "dataset_interface.h"
#include "camera_interface_factory.h"
#include "pinhole_camera.h"

namespace df
{
  namespace drivers
  {

    class HDF5Interface : public DatasetInterface
    {
    public:
      HDF5Interface(const std::string &glob_path);
      virtual ~HDF5Interface();

      virtual void GrabFrames(double &timestamp, cv::Mat *img, cv::Mat *dpt = nullptr) override;
      virtual bool HasMore() override;
      virtual bool HasPoses() override { return true; }
      virtual bool HasIntrinsics() override { return true; }
      virtual cv::Mat GetMask() override {return video_mask_;}

      virtual df::PinholeCamera<float> GetIntrinsics() override { return cam_; }
      virtual std::vector<DatasetFrame> GetAll() override { return std::vector<DatasetFrame>{}; } // unimplemented

    private:
      cv::Mat read_image_from_hdf5(const H5::DataSet &dataset, const H5::DataSpace &dataspace, const int &index, const long &channel);

    private:
      hsize_t count_;
      df::PinholeCamera<float> cam_;
      cv::Mat video_mask_;
      H5::DataSet color_dataset_;
      H5::DataSpace color_dataspace_;
      hsize_t color_dataset_dims_[4];
    };

    class HDF5InterfaceFactory : public SpecificInterfaceFactory
    {
    public:
      virtual std::unique_ptr<CameraInterface> FromUrlParams(const std::string &url_params) override;
      virtual std::string GetUrlPattern(const std::string &prefix_tag) override;
      virtual std::string GetPrefix() override;

    private:
      const std::string url_prefix = "hdf5";
      const std::string url_params = "hdf5_file_path";
    };

  } // namespace drivers
} // namespace df

#endif // DF_HDF5_INTERFACE_H_
