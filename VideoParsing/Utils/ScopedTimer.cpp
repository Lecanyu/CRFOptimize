/**
 * @file ScopedTimer.cpp
 * @brief ScopedTimer
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Utils/ScopedTimer.h"
#include <iostream>

namespace vp {

ScopedTimer::ScopedTimer(const std::string& init_message)
  : beg_(clock_::now()) {
  std::cout << init_message << std::flush;
}

ScopedTimer::~ScopedTimer() {
  std::cout << "Done (" << std::chrono::duration_cast<second_>(clock_::now() - beg_).count() << " s)" << std::endl;
}

}  // namespace vp
