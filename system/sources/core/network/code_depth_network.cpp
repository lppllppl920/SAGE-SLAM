#include "code_depth_network.h"

namespace df
{
	CodeDepthNetwork::CodeDepthNetwork(std::string model_path, int cuda_id = 0)
	{
		try {
			module_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
		}
		catch(c10::Error::exception &error)
		{
			LOG(FATAL) << "[CodeDepthNetwork::CodeDepthNetwork] " << error.what();
		}
		
		module_->eval();
		inputs_.resize(2);
		return;
	}

	CodeDepthNetwork::~CodeDepthNetwork()
	{
	}

	void CodeDepthNetwork::GenerateDepthBiasAndJacobian(const at::Tensor input_image, const at::Tensor input_mask,
																											at::Tensor &dpt_map_bias, at::Tensor &dpt_jac_code)
	{
		inputs_[1] = input_mask;
		inputs_[0] = input_image;

		const std::vector<c10::IValue> outputs = module_->forward(inputs_).toTuple()->elements();

		// 1 x 1 x H x W
		dpt_map_bias = outputs[0].toTensor();
		// 1 x C x H x W
		dpt_jac_code = outputs[1].toTensor();

		const long code_length = dpt_jac_code.size(1);
		// 1 x C x H x W -> C x H*W -> H*W x C
		dpt_jac_code = dpt_jac_code.reshape({code_length, -1}).permute({1, 0});

		return;
	}
} // namespace df
