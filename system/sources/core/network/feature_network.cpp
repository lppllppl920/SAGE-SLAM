#include "feature_network.h"

namespace df
{
	FeatureNetwork::FeatureNetwork(std::string model_path)
	{
		try
		{
			module_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
		}
		catch (c10::Error::exception &error)
		{
			LOG(FATAL) << "[FeatureNetwork::FeatureNetwork] " << error.what();
		}

		module_->eval();
		inputs_.resize(2);
	}

	FeatureNetwork::~FeatureNetwork()
	{
	}

	void FeatureNetwork::GenerateFeatureMaps(const at::Tensor input_image, const at::Tensor input_mask,
																					 at::Tensor &feat_map, at::Tensor &feat_desc)
	{
		inputs_.clear();
		inputs_.resize(2);
		inputs_[1] = input_mask;
		inputs_[0] = input_image;

		// return feature map and descriptor
		auto outputs = module_->forward(inputs_).toTuple()->elements();
		feat_map = outputs[0].toTensor();
		feat_desc = outputs[1].toTensor();
		return;
	}
} // namespace df
