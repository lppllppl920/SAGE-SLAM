#ifndef TENSOR_VOCABULARY_H
#define TENSOR_VOCABULARY_H

#include <vector>
#include <torch/torch.h>
#include <DBoW2/DBoW2.h>

#include "dl_descriptor.h"

namespace df
{

  template <class F>
  class TemplatedTensorVocabulary : public DBoW2::TemplatedVocabulary<at::Tensor, F>
  {
    typedef const at::Tensor *pDescriptor;
    typedef DBoW2::TemplatedVocabulary<at::Tensor, F> Base;

  public:
    /**
   * Initiates an empty vocabulary
   * @param k branching factor
   * @param L depth levels
   * @param weighting weighting type
   * @param scoring scoring type
   */
    TemplatedTensorVocabulary(const std::string &filename, const c10::Device device);

    TemplatedTensorVocabulary(int k = 10,
                              int L = 5,
                              DBoW2::WeightingType weighting =
                                  DBoW2::WeightingType::TF_IDF,
                              DBoW2::ScoringType scoring =
                                  DBoW2::ScoringType::L1_NORM,
                              c10::Device device = c10::Device(torch::kCUDA, 0))
        : DBoW2::TemplatedVocabulary<at::Tensor, F>(k, L, weighting, scoring), m_device(device){};

    void create(const std::vector<std::vector<at::Tensor>> &training_features) override;
    void load(const cv::FileStorage &fs, const std::string &name = "vocabulary") override;
    void transform(const at::Tensor features, DBoW2::BowVector &v) const;
    void setDevice(const c10::Device device) { m_device = device; }

  protected:
    void subset_transform(
        const at::Tensor features, const at::Tensor feature_subset_indexes,
        const long cluster_id, DBoW2::BowVector &v) const;

  protected:
    // C_feat x M
    c10::Device m_device = c10::Device(torch::kCUDA, 0);
    at::Tensor m_node_descriptors;
    std::vector<at::Tensor> m_node_children_indexes;
  };

}

#endif