#include "deepfactors_options.h"

namespace df
{

DeepFactorsOptions::KeyframeMode DeepFactorsOptions::KeyframeModeTranslator(const std::string& s)
{
  if (s == "AUTO")
    return KeyframeMode::AUTO;
  else if (s == "NEVER")
    return KeyframeMode::NEVER;
  else
    LOG(FATAL) << "[DeepFactorsOptions::KeyframeModeTranslator] Unknown KeyframeMode " << s;
    exit(1);
}

std::string DeepFactorsOptions::KeyframeModeTranslator(KeyframeMode mode)
{
  std::string name;
  switch (mode)
  {
  case KeyframeMode::AUTO:
    name = "AUTO";
    break;
  case KeyframeMode::NEVER:
    name = "NEVER";
    break;
  default:
    name = "UNKNOWN";
    break;
  }
  return name;
}

DeepFactorsOptions::TrackingMode DeepFactorsOptions::TrackingModeTranslator(const std::string& s)
{
  if (s == "FIRST")
    return TrackingMode::FIRST;
  else if (s == "LAST")
    return TrackingMode::LAST;
  else if (s == "CLOSEST")
    return TrackingMode::CLOSEST;
  else
    LOG(FATAL) << "[DeepFactorsOptions::TrackingModeTranslator] Invalid tracking mode: " << s;
    exit(1);
}

std::string DeepFactorsOptions::TrackingModeTranslator(TrackingMode mode)
{
  std::string name;
  switch (mode)
  {
  case TrackingMode::FIRST:
    name = "FIRST";
    break;
  case TrackingMode::LAST:
    name = "LAST";
    break;
  case TrackingMode::CLOSEST:
    name = "CLOSEST";
    break;
  default:
    name = "UNKNOWN";
    break;
  }
  return name;
}

} // namespace df
