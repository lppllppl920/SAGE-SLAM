#ifndef DF_CAMERA_INTERFACE_FACTORY_H_
#define DF_CAMERA_INTERFACE_FACTORY_H_

#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>
#include <glog/logging.h>

#include "camera_interface.h"

namespace df
{
namespace drivers
{

/**
 * Exception to indicate that user has passed
 * an URL that does not match accepted pattern URL_PATTERN
 */
class MalformedUrlException : public std::runtime_error
{
public:
  MalformedUrlException(const std::string& pattern, const std::string& reason)
  : std::runtime_error("Invalid source URL " + pattern + ": " + reason) {}
};

/**
 * Base class for a specific interface factory like pointgrey, files, etc
 */
class SpecificInterfaceFactory
{
public:
  virtual std::unique_ptr<CameraInterface> FromUrlParams(const std::string& url_params) = 0;
  virtual std::string GetUrlPattern(const std::string& prefix_tag) = 0;
  virtual std::string GetPrefix() = 0;
};

/**
 * Singleton class that registers our supported supported interfaces
 * and produces CameraInterfaces based on URL
 */
class CameraInterfaceFactory
{
public:
  typedef std::unordered_map<std::string, std::shared_ptr<SpecificInterfaceFactory>> FactoryMapT;

  std::unique_ptr<CameraInterface> GetInterfaceFromUrl(const std::string& url);

  template <typename T>
  void RegisterInterface()
  {
    auto factory_obj = std::make_shared<T>();
    typename FactoryMapT::value_type pair(factory_obj->GetPrefix(), factory_obj);
    factory_map_.insert(pair);
    url_forms_.push_back(factory_obj->GetUrlPattern(prefix_tag_));
    supported_interfaces_.push_back(factory_obj->GetPrefix());
  }

  std::string GetUrlHelp();

  static std::shared_ptr<CameraInterfaceFactory> Get();

private:
  std::vector<std::string> PartitionUrl(const std::string& url);

  FactoryMapT factory_map_;
  std::vector<std::string> supported_interfaces_;
  std::vector<std::string> url_forms_;

  const std::string prefix_tag_ = "://";
  static std::shared_ptr<CameraInterfaceFactory> ptr_;
};

/**
 * Helper class to register new camera interfaces
 * Declare as static variable
 */
template <typename T>
struct InterfaceRegistrar {
  InterfaceRegistrar () {
    CameraInterfaceFactory::Get()->RegisterInterface<T>();
  }
};

} // namespace drivers
} //namespace df

#endif // DF_CAMERA_INTERFACE_FACTORY_H_
