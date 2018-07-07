/**
 * @file VisualizationHelper.h
 * @brief VisualizationHelper
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_UTILS_VISUALIZATIONHELPER_H_
#define VIDEOPARSING_UTILS_VISUALIZATIONHELPER_H_

#include <opencv2/core/core.hpp>
#include <vector>

namespace vp {

class WaitKeyNavigation {
 public:
  WaitKeyNavigation(int lower_bound, int upper_bound, int start_value, bool start_paused = true, bool start_step_fwd = true);
  WaitKeyNavigation(int lower_bound, int upper_bound, bool start_paused = true, bool start_step_fwd = true);

  int value() const {return value_;}
  bool paused() const {return paused_;}
  bool stepFwd() const {return step_fwd_;}

  bool operator()(); // wait and navigate

 private:
  int value_; // Primary value
  const int lower_bound_;
  const int upper_bound_;
  bool paused_;
  bool step_fwd_;
};

void visualizeImages(const std::vector<cv::Mat>& images, const int start_frame,
                     const std::string& window_name = "Images",
                     const int diplay_location = 0, const float scale = 1.0);

void visualizeImages(const std::vector<cv::Mat>& imagesA,
                     const std::vector<cv::Mat>& imagesB,
                     const int start_frame,
                     const std::string& window_name = "Images",
                     const int diplay_location = 0,
                     const float scale = 1.0);


void saveImages(const std::vector<cv::Mat>& images,
                const int start_frame = 0,
                const std::string& out_image_pattern = "SavedImages_%06d.png");


}  // namespace vp


#endif /* VIDEOPARSING_UTILS_VISUALIZATIONHELPER_H_ */
