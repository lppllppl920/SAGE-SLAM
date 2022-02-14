#include <vector>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <DBoW2/DBoW2.h>
#include <boost/filesystem.hpp>
#include <experimental/filesystem>
#include <H5Cpp.h>
#include <regex>
#include <random>
#include <opencv2/imgproc/types_c.h>

#include "feature_network.h"
#include "dl_descriptor.h"
#include "tensor_vocabulary.h"

namespace fs = std::experimental::filesystem;

typedef DBoW2::TemplatedVocabulary<df::FTensor::TDescriptor, df::FTensor>
    TensorVocabulary;
typedef DBoW2::TemplatedDatabase<df::FTensor::TDescriptor, df::FTensor>
    TensorDatabase;

DEFINE_string(data_root, "", "root to hdf5 datasets");
DEFINE_string(output_root, "", "root to the result");
DEFINE_string(pattern, "(.*)(.hdf5)", "hdf5 pattern");
DEFINE_string(feature_network_path, "", "path to feature network trained model");
DEFINE_string(outname, "endoscopy_voc", "Output vocabulary name");
DEFINE_int32(height, 128, "input height");
DEFINE_int32(width, 160, "input width");
DEFINE_int32(out_height, 64, "output height");
DEFINE_int32(out_width, 80, "output width");
DEFINE_int32(max_frames_per_seq, 30, "maximum number of frames per sequence");
DEFINE_int32(k, 10, "Branching factor");
DEFINE_int32(L, 4, "Depth levels");
DEFINE_int32(subsample_points, 1000, "number of samples per image");

// list of paths of all files under the directory 'dir' when the extenstion matches the regex
// file_list<true> searches recursively into sub-directories; file_list<false> searches only the specified directory
template <bool RECURSIVE>
std::vector<fs::path> file_list(fs::path dir, std::regex ext_pattern)
{
  std::vector<fs::path> result;

  using iterator = typename std::conditional<RECURSIVE,
                                             fs::recursive_directory_iterator, fs::directory_iterator>::type;

  const iterator end;
  for (iterator iter{dir}; iter != end; ++iter)
  {
    const std::string extension = iter->path().extension().string();
    if (fs::is_regular_file(*iter) && std::regex_match(extension, ext_pattern))
    {
      result.push_back(*iter);
    }
  }

  return result;
}

cv::Mat read_image_from_hdf5(const H5::DataSet &dataset, const H5::DataSpace &dataspace, const int &index, const long &channel)
{
  hsize_t dims_out[4];
  dataspace.getSimpleExtentNdims();
  dataspace.getSimpleExtentDims(dims_out, NULL);

  hsize_t dims_offset[4] = {static_cast<hsize_t>(index), 0, 0, 0};
  hsize_t dims_expand[4] = {1, dims_out[1], dims_out[2], static_cast<hsize_t>(channel)};
  dataspace.selectHyperslab(H5S_SELECT_SET, dims_expand, dims_offset);

  hsize_t dims_slice[4];
  dims_slice[0] = 1;
  dims_slice[1] = dims_out[1];
  dims_slice[2] = dims_out[2];
  dims_slice[3] = channel;
  H5::DataSpace memspace(4, dims_slice);

  cv::Mat image;
  if (channel == 1)
  {
    image.create((int)dims_out[1], (int)dims_out[2], CV_8UC1);
  }
  else
  {
    image.create((int)dims_out[1], (int)dims_out[2], CV_8UC3);
  }
  dataset.read(image.ptr(0), H5::PredType::NATIVE_UCHAR, memspace, dataspace);

  return image;
}

std::vector<at::Tensor> ChangeStructure(const at::Tensor feat_map, const at::Tensor sampled_1d_locations)
{
  torch::NoGradGuard no_grad;
  using namespace torch::indexing;
  // C x H*W
  long channel = feat_map.size(0);
  long num_points = sampled_1d_locations.size(0);

  // sampled location only?
  std::vector<at::Tensor> features(num_points, torch::zeros({channel}, feat_map.options()));
  long index;
  for (int i = 0; i < num_points; ++i)
  {
    index = sampled_1d_locations.index({i}).item<long>();
    features[i] = feat_map.index({Slice(), index});
  }

  return features;
}

void generate_samples(const at::Tensor locations_1d, const long &num_points, at::Tensor &sample_locations_1d, const long seed)
{

  std::vector<long> indices(locations_1d.size(0));
  std::iota(indices.begin(), indices.end(), 0);
  // shuffle and take the first set of indices as the randomly selected ones
  // std::random_device rd;
  std::mt19937 g; // (rd())
  g.seed(seed);
  std::shuffle(indices.begin(), indices.end(), g);

  const at::Tensor &sample_indexes = torch::from_blob(static_cast<long *>(indices.data()),
                                                      {num_points}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
                                         .to(locations_1d.device())
                                         .clone();
  // K
  sample_locations_1d = locations_1d.index({sample_indexes});
}

int main(int argc, char *argv[])
{
  LOG(INFO) << "Starting voc_builder";
  torch::NoGradGuard no_grad;

  // init google logging
  // google::LogToStderr();
  google::InitGoogleLogging(argv[0]);

  // parse command line flags
  google::SetUsageMessage("Tensor Vocabulary Builder");
  google::ParseCommandLineFlags(&argc, &argv, true);

  FLAGS_alsologtostderr = true;
  FLAGS_stderrthreshold = 0;
  FLAGS_minloglevel = 0;
  FLAGS_log_dir = "";
  FLAGS_v = 0;

  std::fstream fp;
  fp.open(FLAGS_log_dir, std::fstream::in | std::fstream::out | std::fstream::app);

  if (!fp)
  {
    fp.open(FLAGS_log_dir, std::fstream::in | std::fstream::out | std::fstream::trunc);
    fp.close();
  }

  std::vector<fs::path> hdf5_path_list = file_list<true>(fs::path(FLAGS_data_root), std::regex(FLAGS_pattern));
  df::FeatureNetwork feature_network{FLAGS_feature_network_path};

  // create dataset interface
  std::vector<std::vector<at::Tensor>> features;
  cv::Mat color, mask, resized_color, resized_mask, out_resized_mask;

  at::Tensor input_color, input_mask, output_mask;
  at::Tensor feat_map, feat_desc, sampled_1d_locations, subsampled_1d_locations;

  long feat_count = 0;
  for (size_t i = 0; i < hdf5_path_list.size(); ++i)
  {
    LOG(INFO) << "[voc_builder] Processing " << hdf5_path_list[i].c_str();
    H5::H5File hdf5_file(hdf5_path_list[i].c_str(), H5F_ACC_RDONLY);

    // read mask image per sequence
    H5::DataSet mask_dataset = hdf5_file.openDataSet("mask");
    H5::DataSpace mask_dataspace = mask_dataset.getSpace();
    hsize_t mask_dims_out[4];
    mask_dataspace.getSimpleExtentDims(mask_dims_out, NULL);
    mask = read_image_from_hdf5(mask_dataset, mask_dataspace, 0, 1);
    cv::resize(mask, resized_mask, cv::Size(FLAGS_width, FLAGS_height), 0, 0, CV_INTER_NN);
    cv::resize(mask, out_resized_mask, cv::Size(FLAGS_out_width, FLAGS_out_height), 0, 0, CV_INTER_NN);

    output_mask = torch::from_blob(static_cast<unsigned char *>(out_resized_mask.data),
                                   {1, 1, FLAGS_out_height, FLAGS_out_width},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                      .to(torch::kCUDA, 0)
                      .to(torch::kFloat32)
                      .clone();
    input_mask = torch::from_blob(static_cast<unsigned char *>(resized_mask.data),
                                  {1, 1, FLAGS_height, FLAGS_width},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                     .to(torch::kCUDA, 0)
                     .to(torch::kFloat32)
                     .clone();

    sampled_1d_locations = torch::nonzero(output_mask.reshape({-1}) > 0.9).reshape({-1});

    // read color video frames
    H5::DataSet color_dataset = hdf5_file.openDataSet("color");
    H5::DataSpace color_dataspace = color_dataset.getSpace();
    hsize_t color_dims_out[4];
    color_dataspace.getSimpleExtentDims(color_dims_out, NULL);

    const long interval = static_cast<long>(std::min(1.0f, static_cast<float>(color_dims_out[0]) / (float)FLAGS_max_frames_per_seq));

    for (long j = 0; j < std::min((long)FLAGS_max_frames_per_seq * interval, static_cast<long>(color_dims_out[0])); j += interval)
    {
      generate_samples(sampled_1d_locations, FLAGS_subsample_points, subsampled_1d_locations, j);

      color = read_image_from_hdf5(color_dataset, color_dataspace, j, 3);
      cv::resize(color, resized_color, cv::Size(FLAGS_width, FLAGS_height), 0, 0, CV_INTER_AREA);

      input_color = torch::from_blob(static_cast<unsigned char *>(resized_color.data),
                                     {1, static_cast<long>(FLAGS_height), static_cast<long>(FLAGS_width), 3},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                        .to(torch::kCUDA, 0)
                        .to(torch::kFloat32)
                        .permute({0, 3, 1, 2})
                        .clone() /
                    255.0f;

      feature_network.GenerateFeatureMaps(input_color, input_mask, feat_map, feat_desc);
      std::vector<at::Tensor> feature_vec = ChangeStructure(feat_desc.to(torch::kCPU).reshape({feat_desc.size(1), -1}), subsampled_1d_locations);
      features.push_back(feature_vec);
      feat_count += feature_vec.size();
    }
  }

  // build vocabulary
  TensorVocabulary voc(FLAGS_k, FLAGS_L, DBoW2::TF_IDF, DBoW2::L1_NORM);

  LOG(INFO) << "[voc_builder]Creating vocabulary using " << features.size() << " images that contains in total " << feat_count << " features";
  voc.create(features);
  LOG(INFO) << "[voc_builder] Done";

  fs::path output_path = fs::path(FLAGS_output_root + "/");
  output_path += fs::path(FLAGS_outname);
  voc.save(std::string(output_path.c_str()) + ".yml.gz");
}
