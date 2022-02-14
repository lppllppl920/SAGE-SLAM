#include "tensor_vocabulary.h"

namespace df
{

  template <class F>
  TemplatedTensorVocabulary<F>::TemplatedTensorVocabulary(const std::string &filename, const c10::Device device)
  {
    m_device = device;
    this->m_scoring_object = NULL;
    cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
    if (!fs.isOpened())
    {
      throw std::runtime_error(std::string("Could not open file ") + filename);
    }

    load(fs);
  }

  template <class F>
  void TemplatedTensorVocabulary<F>::create(const std::vector<std::vector<at::Tensor>> &training_features)
  {
    this->m_nodes.clear();
    this->m_words.clear();

    // expected_nodes = Sum_{i=0..L} ( k^i )
    int expected_nodes =
        (int)((pow((double)this->m_k, (double)this->m_L + 1) - 1) / (this->m_k - 1));

    this->m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree

    std::vector<pDescriptor> features;
    Base::getFeatures(training_features, features);

    // create root
    this->m_nodes.push_back(typename Base::Node(0)); // root

    // create the tree
    Base::HKmeansStep(0, features, 1);

    // create the words
    Base::createWords();

    // and set the weight of each node of the tree
    Base::setNodeWeights(training_features);
  }

  template <class F>
  void TemplatedTensorVocabulary<F>::load(const cv::FileStorage &fs,
                                          const std::string &name)
  {
    using namespace torch::indexing;
    this->m_words.clear();
    this->m_nodes.clear();

    cv::FileNode fvoc = fs[name];

    this->m_k = (int)fvoc["k"];
    this->m_L = (int)fvoc["L"];
    this->m_scoring = (DBoW2::ScoringType)((int)fvoc["scoringType"]);
    this->m_weighting = (DBoW2::WeightingType)((int)fvoc["weightingType"]);

    Base::createScoringObject();

    // nodes
    cv::FileNode fn = fvoc["nodes"];

    std::vector<at::Tensor> node_descriptors;
    node_descriptors.resize(fn.size() + 1);

    this->m_nodes.resize(fn.size() + 1);
    this->m_nodes[0].id = 0;

    m_node_children_indexes = std::vector<at::Tensor>(static_cast<long>(this->m_nodes.size()),
                                                      torch::zeros({1}, torch::TensorOptions().device(m_device).dtype(torch::kLong)));

    for (unsigned int i = 0; i < fn.size(); ++i)
    {
      DBoW2::NodeId nid = static_cast<int>(fn[i]["nodeId"]);
      DBoW2::NodeId pid = static_cast<int>(fn[i]["parentId"]);
      DBoW2::WordValue weight = static_cast<DBoW2::WordValue>(fn[i]["weight"]);
      std::string d = (std::string)fn[i]["descriptor"];

      this->m_nodes[nid].id = nid;
      this->m_nodes[nid].parent = pid;
      this->m_nodes[nid].weight = weight;
      this->m_nodes[pid].children.push_back(nid);

      // Store the at::Tensor in a std vector to make it easier to be indexed later
      F::fromString(node_descriptors[nid], d, m_device);
    }

    // The descriptor of root node above was undefined, fill it in here
    node_descriptors[0] = torch::zeros_like(node_descriptors[1]);

    // Load all data to GPU instead CPU
    // Construct node children indexes as at::Tensor
    for (long i = 0; i < static_cast<long>(this->m_nodes.size()); ++i)
    {
      // Be careful not to delete children etc later
      if (!this->m_nodes[i].children.empty())
      {
        // static_cast<long *>(
        const at::Tensor temp = torch::from_blob(this->m_nodes[i].children.data(),
                                                  {static_cast<long>(this->m_nodes[i].children.size())},
                                                  torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                                     .to(m_device);
        m_node_children_indexes[i] = temp;
      }
    }

    // C_feat x N
    m_node_descriptors = torch::stack(node_descriptors, 1);

    // words
    fn = fvoc["words"];

    this->m_words.resize(fn.size());

    for (unsigned int i = 0; i < fn.size(); ++i)
    {
      DBoW2::NodeId wid = (int)fn[i]["wordId"];
      DBoW2::NodeId nid = (int)fn[i]["nodeId"];

      this->m_nodes[nid].word_id = wid;
      this->m_words[wid] = &(this->m_nodes[nid]);
    }
  }

  template <class F>
  void TemplatedTensorVocabulary<F>::transform(
      const at::Tensor features, typename DBoW2::BowVector &v) const
  {
    using namespace DBoW2;
    // Currently we only support these two modes
    assert(this->m_weighting == WeightingType::TF || this->m_weighting == WeightingType::TF_IDF);

    using namespace torch::indexing;
    v.clear();

    if (this->m_words.empty())
    {
      return;
    }

    const long feat_channel = features.size(0);
    const long num_points = features.size(1);

    // normalize
    typename DBoW2::LNorm norm;
    bool must = this->m_scoring_object->mustNormalize(norm);

    const at::Tensor children_indexes = m_node_children_indexes[0];

    // C_feat x K
    const at::Tensor children_descriptors = m_node_descriptors.index({Slice(), children_indexes});
    const long children_size = children_descriptors.size(1);

    // C_feat x N x K -> N x K
    // Here assumes the F class uses L2-norm distance as distance definition
    const at::Tensor distances = torch::sum(torch::square(features.reshape({feat_channel, num_points, 1}) -
                                                           children_descriptors.reshape({feat_channel, 1, children_size})),
                                             0, false);

    // Get minimum-distance cluster id for each descriptor
    // N
    const at::Tensor cluster_ids = torch::argmin(distances, 1, false).to(torch::kLong);
    for (long i = 0; i < children_size; ++i)
    {
      // S
      const at::Tensor feature_subset_indexes = torch::nonzero(cluster_ids == i).reshape({-1});
      if (feature_subset_indexes.size(0) >= 1)
      {
        subset_transform(features, feature_subset_indexes, children_indexes.index({i}).item<long>(), v);
      }
    }

    if (!v.empty() && !must)
    {
      // unnecessary when normalizing
      const double nd = v.size();
      for (DBoW2::BowVector::iterator vit = v.begin(); vit != v.end(); vit++)
        vit->second /= nd;
    }

    if (must)
    {
      v.normalize(norm);
    }

    return;
  }

  // The goal is to reach the wordId level and fill in the corresponding location of BoWVector, nothing needs to be returned (?)
  template <class F>
  void TemplatedTensorVocabulary<F>::subset_transform(
      const at::Tensor features, const at::Tensor feature_subset_indexes,
      const long cluster_id, DBoW2::BowVector &v) const
  {
    // ending condition: reaching leaf node
    if (this->m_nodes[cluster_id].children.empty())
    {
      // modify the BoWVector to add corresponding weight value
      const DBoW2::WordValue weight = this->m_nodes[cluster_id].weight;
      if (weight > 0)
      {
        v.addWeight(this->m_nodes[cluster_id].word_id, weight * features.size(1));
      }

      return;
    }

    using namespace torch::indexing;
    // K

    const at::Tensor children_indexes = m_node_children_indexes[cluster_id];
    // C_feat x K
    const at::Tensor children_descriptors = m_node_descriptors.index({Slice(), children_indexes});
    const long children_size = children_descriptors.size(1);

    const at::Tensor subset_features = features.index({Slice(), feature_subset_indexes});
    const long feat_channel = subset_features.size(0);
    const long num_subset_points = subset_features.size(1);
    // C_feat x N x K -> N x K
    // Here assumes the F class uses L2-norm distance as distance definition
    const at::Tensor distances = torch::sum(torch::square(subset_features.reshape({feat_channel, num_subset_points, 1}) -
                                                           children_descriptors.reshape({feat_channel, 1, children_size})),
                                             0, false);

    // Get minimum-distance cluster id for each descriptor
    // N
    const at::Tensor cluster_ids = torch::argmin(distances, 1, false).to(torch::kLong);
    for (long i = 0; i < children_size; ++i)
    {
      // S
      const at::Tensor feature_indexes_within_subset = torch::nonzero(cluster_ids == i).reshape({-1});
      if (feature_indexes_within_subset.size(0) >= 1)
      {
        subset_transform(features, feature_subset_indexes.index({feature_indexes_within_subset}),
                         children_indexes.index({i}).item<long>(), v);
      }
    }

    return;
  }

  template class TemplatedTensorVocabulary<FTensor>;
}
