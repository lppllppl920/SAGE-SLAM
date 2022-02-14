#ifndef DF_TIMING_H_
#define DF_TIMING_H_

#include <ctime>
#include <string>

void EnableTiming(bool enable);
void tic(std::string name);
void toc(std::string name);

template <typename Func>
double MeasureTime(Func f)
{
  auto start = std::clock();
  f();
  return (std::clock()-start) / (double)( CLOCKS_PER_SEC / 1000);
}

template <typename Func>
double MeasureTimeAverage(Func f, long ntests, bool skip_first=true)
{
  if (skip_first)
    MeasureTime(f);
  double total = 0;
  for (long i = 0; i < ntests; ++i)
    total += MeasureTime(f);
  return total / ntests;
}

#endif // DF_TIMING_H_
