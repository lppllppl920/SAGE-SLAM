#ifndef FEATURENETWORK_H_
#define FEATURENETWORK_H_

#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <string>

namespace df
{
	class FeatureNetwork
	{

	public:
		typedef std::shared_ptr<FeatureNetwork> Ptr;
		typedef at::Tensor Tensor;
		FeatureNetwork(){};
		FeatureNetwork(std::string model_path);
		virtual ~FeatureNetwork();
		void GenerateFeatureMaps(const at::Tensor input_image, const at::Tensor input_mask, Tensor &feat_map, Tensor &feat_desc);

	private:
		std::shared_ptr<torch::jit::script::Module> module_;
		std::vector<c10::IValue> inputs_;
	};
} // namespace df
#endif /* FEATURENETWORK_H_ */
