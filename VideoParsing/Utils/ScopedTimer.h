/**
 * @file ScopedTimer.h
 * @brief ScopedTimer
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_UTILS_SCOPEDTIMER_H_
#define VIDEOPARSING_UTILS_SCOPEDTIMER_H_

#include <chrono>
#include <string>

namespace vp {

class ScopedTimer {
 public:
  ScopedTimer(const std::string& init_message = "Timer started ...");
  ~ScopedTimer();

 private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1> > second_;
  std::chrono::time_point<clock_> beg_;
};

}  // namespace vp

#endif /* VIDEOPARSING_UTILS_SCOPEDTIMER_H_ */
