#ifndef DF_DISPLAY_UTILS_H_
#define DF_DISPLAY_UTILS_H_

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

cv::Mat apply_colormap(cv::Mat mat, cv::ColormapTypes cmap = cv::COLORMAP_JET);

cv::Mat CreateMosaic(const std::vector<cv::Mat>& array, int rows, int cols);

cv::Mat CreateMosaic(const std::vector<cv::Mat>& array);

cv::Mat Tensor2Mat(const at::Tensor tensor);

#endif // DF_DISPLAY_UTILS_H_
