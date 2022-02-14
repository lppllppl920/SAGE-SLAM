#include "dl_descriptor.h"

namespace df
{

  void FTensor::meanValue(const std::vector<FTensor::pDescriptor> &descriptors,
                          FTensor::TDescriptor &mean)
  {
    torch::NoGradGuard no_grad;

    mean = torch::zeros_like(*(descriptors[0]));
    long num_points = descriptors.size();
    for (long i = 0; i < num_points; ++i)
    {
      mean += *(descriptors[i]);
    }

    mean /= (float)num_points;

    return;
  }

  double FTensor::distance(const FTensor::TDescriptor &a, const FTensor::TDescriptor &b)
  {
    torch::NoGradGuard no_grad;
    const at::Tensor norm = torch::norm(a - b, 2);
    return (double)(norm.item<float>());
  }

  std::string FTensor::toString(const FTensor::TDescriptor &a)
  {
    torch::NoGradGuard no_grad;

    std::vector<float> a_vec(a.size(0));
    std::memcpy(static_cast<float *>(a_vec.data()), a.to(torch::kFloat32).contiguous().data_ptr(), sizeof(float) * a.numel());

    std::stringstream ss;
    std::copy(a_vec.begin(), a_vec.end(), std::ostream_iterator<float>(ss, " "));
    return ss.str();
  }

  void FTensor::fromString(FTensor::TDescriptor &a, const std::string &s, const c10::Device& device)
  {
    torch::NoGradGuard no_grad;
    std::istringstream iss(s);
    std::vector<float> results;
    std::copy(std::istream_iterator<float>{iss},
              std::istream_iterator<float>(), std::back_inserter(results));
    a = torch::from_blob(results.data(), {static_cast<long>(results.size())},
                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
            .clone()
            .to(device);
    return;
  }

  void FTensor::toMat32F(const std::vector<TDescriptor> &descriptors,
                         cv::Mat &mat)
  {
    torch::NoGradGuard no_grad;
    if (descriptors.empty())
    {
      mat.release();
      return;
    }

    const int N = descriptors.size();
    const int L = descriptors[0].size(0);

    mat.create(N, L, CV_32F);

    for (int i = 0; i < N; ++i)
    {
      const TDescriptor &desc = descriptors[i];
      float *p = mat.ptr<float>(i);
      std::memcpy(static_cast<float *>(p), desc.to(torch::kCPU).to(torch::kFloat32).contiguous().data_ptr(), sizeof(float) * desc.numel());
    }

    return;
  }
} // namespace df