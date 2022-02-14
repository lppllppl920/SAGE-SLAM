#ifndef DF_LOGUTILS_H_
#define DF_LOGUTILS_H_

#include <ctime>
#include <string>
#include <boost/filesystem.hpp>

namespace df
{

std::string GetTimeStamp(std::string format = "%y%m%d%H%M%S");
void CreateDirIfNotExists(std::string dir);
std::string CreateLogDirForRun(std::string logdir, std::string run_dir_name);

} // namespace df

#endif // DF_LOGUTILS_H_
