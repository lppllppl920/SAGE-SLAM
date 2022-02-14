#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>

#include "logutils.h"

namespace fs = boost::filesystem;

namespace df
{

std::string GetTimeStamp(std::string format)
{
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d-%X");
  return ss.str();
}

void CreateDirIfNotExists(std::string dir)
{
  if (!fs::exists(dir))
    fs::create_directory(dir);
}

std::string CreateLogDirForRun(std::string logdir, std::string run_dir_name)
{
  // check if the logging dir exists. If not, create it
  CreateDirIfNotExists(logdir);

  std::string dir_name = run_dir_name;
  std::string rundir = (fs::path(logdir) / dir_name).string();

  // This directory shouldn't exist
  CreateDirIfNotExists(rundir); 

  return rundir;
}

} // namespace df
