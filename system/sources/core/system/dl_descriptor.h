#ifndef __D_T_F_DLDESC__
#define __D_T_F_DLDESC__

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <sstream>
#include <DBoW2/DBoW2.h>

namespace df
{
  /// Functions to manipulate Torch::Tensor descriptors
  class FTensor : protected DBoW2::FClass
  {
  public:
    /// Descriptor type
    typedef at::Tensor TDescriptor;
    /// Pointer to a single descriptor
    typedef const TDescriptor *pDescriptor;

    /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors vector of pointers to descriptors
   * @param mean mean descriptor
   */
    static void meanValue(const std::vector<pDescriptor> &descriptors,
                          TDescriptor &mean);

    /**
   * Calculates the (squared) distance between two descriptors
   * @param a
   * @param b
   * @return (squared) distance
   */
    static double distance(const TDescriptor &a, const TDescriptor &b);

    /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
    static std::string toString(const TDescriptor &a);

    /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
    // static void fromString(TDescriptor &a, const std::string &s);
    static void fromString(TDescriptor &a, const std::string &s, const c10::Device& device = c10::Device(torch::kCUDA, 0)); // , const c10::Device device = c10::Device(torch::kCPU)

    /**
   * Returns a mat with the descriptors in float format
   * @param descriptors
   * @param mat (out) NxL 32F matrix
   */
    static void toMat32F(const std::vector<TDescriptor> &descriptors,
                         cv::Mat &mat);
  };

} // namespace df

#endif