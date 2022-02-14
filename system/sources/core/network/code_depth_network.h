#ifndef CODEDEPTHNETWORK_H_
#define CODEDEPTHNETWORK_H_

#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <string>

#include "timing.h"
namespace df
{
	class CodeDepthNetwork
	{
	public:
		typedef std::shared_ptr<CodeDepthNetwork> Ptr;
		typedef at::Tensor Tensor;
		CodeDepthNetwork(std::string model_path, int cuda_id);
		virtual ~CodeDepthNetwork();

		void GenerateDepthBiasAndJacobian(const at::Tensor input_image, const at::Tensor input_mask,
																			at::Tensor &dpt_map_bias, at::Tensor &dpt_jac_code);

	private:
		std::shared_ptr<torch::jit::script::Module> module_;
		std::vector<torch::jit::IValue> inputs_;
	};
} // namespace df

#endif /* CODEDEPTHNETWORK_H_ */
