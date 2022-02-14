#include <glog/logging.h>
#include "display_utils.h"

cv::Mat apply_colormap(cv::Mat mat, cv::ColormapTypes cmap)
{
  cv::Mat ret;
  mat.convertTo(ret, CV_8UC1, 255.0);
  cv::applyColorMap(ret, ret, cmap);
  return ret;
}

// cv::Mat GetOpenCV(const vc::Buffer2DView<float, vc::TargetHost> &buf)
// {
//   return cv::Mat(buf.height(), buf.width(), CV_32FC1, buf.rawPtr());
// }

cv::Mat Tensor2Mat(const at::Tensor tensor)
{
  int64_t channel = tensor.size(1);
  int64_t height = tensor.size(2);
  int64_t width = tensor.size(3);
  cv::Mat mat(height, width, CV_32FC(channel));
  std::memcpy(static_cast<float *>(mat.ptr<float>(0)),
              tensor.permute({0, 2, 3, 1}).to(torch::kFloat32).to(torch::kCPU).contiguous().data_ptr(),
              sizeof(float) * tensor.numel());

  return mat;
}
//cv::Mat prx_to_color(vc::Buffer2DView<float, vc::TargetHost> prxbuf, cv::ColormapTypes cmap)
//{
//  return apply_colormap(GetOpenCV(prxbuf), cmap);
//}

/**
* @brief CreateMosaic
* Creates a mosaic that tiles images from the input array in a specified shape.
* This function attempts to unify all input images. All images are resized to match
* the first array element.
* @param array A linear vector of input images
* @param rows Number of rows
* @param cols
* @return CV_8UC3 mosaic image
*/
cv::Mat CreateMosaic(const std::vector<cv::Mat> &array, int rows, int cols)
{
  cv::Mat mosaic;

  if (array.empty())
    return mosaic;

  int width = array[0].cols;
  int height = array[0].rows;

  mosaic.create(rows * height, cols * width, CV_8UC3);

  int idx = 0;
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      cv::Mat img = array[idx++];

      // resize input image to match the first image in array
      if (width != img.cols || height != img.rows)
        cv::resize(img, img, cv::Size(width, height));

      cv::Mat img_disp;
      switch (img.type())
      {
      case CV_8UC3:
        img_disp = img;
        break;
      case CV_8UC1:
        cv::cvtColor(img, img_disp, cv::COLOR_GRAY2BGR);
        break;
      case CV_32FC1:
        cv::cvtColor(img, img_disp, cv::COLOR_GRAY2BGR);
        img_disp.convertTo(img_disp, CV_8UC3, 255.0);
        break;
      default:
        LOG(FATAL) << "[CreateMosaic] Unexpected mat type: " << img.type();
      }

      cv::Rect roi(j * width, i * height, width, height);
      cv::Mat dst = mosaic(roi);
      img_disp.copyTo(dst);
    }
  }

  return mosaic;
}

cv::Mat CreateMosaic(const std::vector<cv::Mat> &array)
{
  int cols = (int)sqrt(array.size());
  int rows = array.size() / cols;
  return CreateMosaic(array, rows, cols);
}
