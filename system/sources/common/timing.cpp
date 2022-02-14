#include <ctime>
#include <mutex>
#include <map>
#include <glog/logging.h>

std::map<std::string, std::clock_t> __clocks;
std::mutex clocks_mutex;
bool enable_timing = false;

void EnableTiming(bool enable)
{
  enable_timing = enable;
}

void tic(std::string name)
{
  if (enable_timing)
  {
#ifdef ENABLE_TIMING
    std::lock_guard<std::mutex> lock(clocks_mutex);
    if (__clocks.find(name) != __clocks.end())
      VLOG(1) << "[tic] Warning: restarting an existing clock (" << name << ")";
    __clocks[name] = std::clock();
#else
    LOG(FATAL) << "[tic] Requesting enable_timing but it was disabled during compilation!";
#endif
  }
}

void toc(std::string name)
{
  if (enable_timing)
  {
#ifdef ENABLE_TIMING
    std::lock_guard<std::mutex> lock(clocks_mutex);
    if (__clocks.find(name) == __clocks.end())
      LOG(FATAL) << "[toc] Trying to use measure a clock that was not started with tic (" << name << ")";

    // print and remove from clocks
    double elapsed_ms = double(std::clock() - __clocks[name]) / CLOCKS_PER_SEC * 1000.0f;
    VLOG(1) << name << " time: " << elapsed_ms;

    __clocks.erase(name);
#else
    LOG(FATAL) << "[toc] Requesting enable_timing but it was disabled during compilation!";
#endif
  }
}
