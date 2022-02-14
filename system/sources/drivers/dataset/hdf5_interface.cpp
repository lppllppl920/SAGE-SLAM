#include "hdf5_interface.h"

namespace df
{
  namespace drivers
  {
    static InterfaceRegistrar<HDF5InterfaceFactory> automatic;

    HDF5Interface::HDF5Interface(const std::string &hdf5_path) : count_(0)
    {
      H5::H5File hdf5_file;
      try
      {
        hdf5_file.openFile(hdf5_path, H5F_ACC_RDONLY);
      }
      catch (H5::Exception &error)
      {
        LOG(FATAL) << "[HDF5Interface::HDF5Interface] " << error.getDetailMsg();
      }

      // Read video mask (assuming only one video mask for the entire video)
      H5::DataSet mask_dataset = hdf5_file.openDataSet("mask");
      H5::DataSpace mask_dataspace = mask_dataset.getSpace();
      hsize_t mask_dims_out[4];
      mask_dataspace.getSimpleExtentDims(mask_dims_out, NULL);

      video_mask_ = read_image_from_hdf5(mask_dataset, mask_dataspace, 0, 1);

      // Read camera intrinsics
      H5::DataSet cam_dataset = hdf5_file.openDataSet("intrinsics");
      H5::DataSpace cam_dataspace = cam_dataset.getSpace();
      hsize_t cam_dims_out[3];
      cam_dataspace.getSimpleExtentDims(cam_dims_out, NULL);

      hsize_t dims_offset[3] = {0, 0, 0};
      hsize_t dims_expand[4] = {1, cam_dims_out[1], cam_dims_out[2]};
      cam_dataspace.selectHyperslab(H5S_SELECT_SET, dims_expand, dims_offset);

      hsize_t cam_dims_slice[3];
      cam_dims_slice[0] = 1;
      cam_dims_slice[1] = cam_dims_out[1];
      cam_dims_slice[2] = cam_dims_out[2];
      H5::DataSpace cam_memspace(3, cam_dims_slice);

      std::vector<float> cam_vector(9);
      cam_dataset.read(cam_vector.data(), H5::PredType::NATIVE_FLOAT, cam_memspace, cam_dataspace);

      // fx, fy, cx, cy, width, height
      cam_ = df::PinholeCamera<float>(cam_vector[0], cam_vector[4], cam_vector[2], cam_vector[5], video_mask_.cols, video_mask_.rows);

      // Load color dataset and dataspace
      color_dataset_ = hdf5_file.openDataSet("color");
      color_dataspace_ = color_dataset_.getSpace();
      color_dataspace_.getSimpleExtentDims(color_dataset_dims_, NULL);
    }

    HDF5Interface::~HDF5Interface()
    {
    }

    cv::Mat HDF5Interface::read_image_from_hdf5(const H5::DataSet &dataset, const H5::DataSpace &dataspace, const int &index, const long &channel)
    {
      hsize_t dims_out[4];
      dataspace.getSimpleExtentDims(dims_out, NULL);

      hsize_t dims_offset[4] = {static_cast<hsize_t>(index), 0, 0, 0};
      hsize_t dims_expand[4] = {1, dims_out[1], dims_out[2], static_cast<hsize_t>(channel)};
      dataspace.selectHyperslab(H5S_SELECT_SET, dims_expand, dims_offset);

      hsize_t dims_slice[4];
      dims_slice[0] = 1;
      dims_slice[1] = dims_out[1];
      dims_slice[2] = dims_out[2];
      dims_slice[3] = channel;
      H5::DataSpace memspace(4, dims_slice);

      cv::Mat image;
      if (channel == 1)
      {
        image.create((int)dims_out[1], (int)dims_out[2], CV_8UC1);
      }
      else
      {
        image.create((int)dims_out[1], (int)dims_out[2], CV_8UC3);
      }
      dataset.read(image.ptr(0), H5::PredType::NATIVE_UCHAR, memspace, dataspace);

      return image;
    }

    void HDF5Interface::GrabFrames(double &timestamp, cv::Mat *img, cv::Mat *dpt)
    {
      if (dpt != nullptr)
      {
        throw UnsupportedFeatureException("Depth not supported by this camera");
      }

      timestamp = static_cast<double>(count_);
      if (count_ < color_dataset_dims_[0] - 1)
      {
        count_++;
      }

      *img = read_image_from_hdf5(color_dataset_, color_dataspace_, count_, 3);

      return;
    }

    bool HDF5Interface::HasMore()
    {
      return count_ <= color_dataset_dims_[0] - 2;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Factory class for this interface
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::unique_ptr<CameraInterface> HDF5InterfaceFactory::FromUrlParams(const std::string &url_params)
    {
      // we expect url_params to be a hdf5 file path
      return std::make_unique<HDF5Interface>(url_params);
    }

    std::string HDF5InterfaceFactory::GetUrlPattern(const std::string &prefix_tag)
    {
      return url_prefix + prefix_tag + url_params;
    }

    std::string HDF5InterfaceFactory::GetPrefix()
    {
      return url_prefix;
    }

  }

}