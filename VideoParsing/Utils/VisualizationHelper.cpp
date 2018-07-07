/**
 * @file VisualizationHelper.cpp
 * @brief VisualizationHelper
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Utils/VisualizationHelper.h"

#include <opencv2/highgui/highgui.hpp>
#include <boost/format.hpp>
#include <iostream>

namespace vp {

WaitKeyNavigation::WaitKeyNavigation(int lower_bound, int upper_bound, int start_value, bool start_paused, bool start_step_fwd)
    : value_(start_value),
      lower_bound_(lower_bound),
      upper_bound_(upper_bound),
      paused_(start_paused),
      step_fwd_(start_step_fwd) {
  value_ = std::max(lower_bound_, value_);
  value_ = std::min(value_, upper_bound_);

  std::cout << "Use W/A/S/D to move fwd/bwd. Use P to toggle pause/unpause.\n";
}

WaitKeyNavigation::WaitKeyNavigation(int lower_bound, int upper_bound, bool start_paused, bool start_step_fwd)
    : value_(lower_bound),
      lower_bound_(lower_bound),
      upper_bound_(upper_bound),
      paused_(start_paused),
      step_fwd_(start_step_fwd){
  value_ = std::max(lower_bound_, value_);
  value_ = std::min(value_, upper_bound_);
  std::cout << "Use W/A/S/D to move fwd/bwd. Use P to toggle pause/unpause.\n";
}

bool WaitKeyNavigation::operator ()() {

  const int key = cv::waitKey(!paused_);

  if (key == 27 || key == 81 || key == 113)  // Esc or Q or q
    return false;
  else if (key == 123 || key == 125 || key == 50 || key == 52 || key == 'a' || key == 'A'
      || key == 's' || key == 'S')  // Down or Left Arrow key (including numpad) or 'a' and 's'
    step_fwd_ = false;
  else if (key == 124 || key == 126 || key == 54 || key == 56 || key == 'w' || key == 'W'
      || key == 'd' || key == 'D')  // Up or Right Arrow or 'w' or 'd'
    step_fwd_ = true;
  else if (key == 'p' || key == 'P')
    paused_ = !paused_;

  if(step_fwd_)
    ++value_;
  else
    --value_;

  value_ = std::max(lower_bound_, value_);
  value_ = std::min(value_, upper_bound_);

  return true;
}

void visualizeImages(const std::vector<cv::Mat>& images,
                     const int start_frame,
                     const std::string& window_name,
                     const int diplay_location,
                     const float scale) {

  const int last_image_id = images.size() - 1;

  cv::namedWindow(window_name, CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  cv::moveWindow(window_name, diplay_location * images.front().cols, 0);
  cv::resizeWindow(window_name, images.front().cols * scale, images.front().rows * scale);

  const boost::format window_msg(" Frame = %d/%d   Paused = %s");

  // Now display the results until q or Esc is pressed

  WaitKeyNavigation nav(0, last_image_id, true, true);
  do {
    const int i = nav.value();
    const std::string overlay_msg = (boost::format(window_msg) % (i + start_frame) % (last_image_id + start_frame)
        % (nav.paused() ? "ON" : "OFF")).str();
    cv::displayOverlay(window_name,overlay_msg);
    cv::imshow(window_name, images.at(i));
  } while(nav());
}

void visualizeImages(const std::vector<cv::Mat>& imagesA,
                     const std::vector<cv::Mat>& imagesB,
                     const int start_frame,
                     const std::string& window_name,
                     const int diplay_location,
                     const float scale) {
  if(imagesA.size() != imagesB.size())
    throw std::runtime_error("imagesA.size() != imagesB.size()");

  std::vector<cv::Mat> canvas_images(imagesA.size());
  const int mid_border = 5;
#pragma omp parallel for
  for(int i = 0; i < canvas_images.size(); ++i) {
    const cv::Mat& imgA = imagesA[i];
    const cv::Mat& imgB = imagesB[i];

    if(imgA.type() != CV_8UC3)
      throw std::runtime_error("Only supports CV_8UC3");

    if(imgB.type() != CV_8UC3)
      throw std::runtime_error("Only supports CV_8UC3");

    canvas_images[i] = cv::Mat::zeros(imgA.rows + mid_border + imgB.rows, std::max(imgA.cols, imgB.cols), CV_8UC3 );

    imgA.copyTo(canvas_images[i](cv::Range(0,imgA.rows),cv::Range(0,imgA.cols)));
    imgB.copyTo(canvas_images[i](cv::Range(imgA.rows + mid_border, imgA.rows + mid_border + imgB.rows),cv::Range(0,imgB.cols)));
  }


  visualizeImages(canvas_images, start_frame,window_name, diplay_location, scale);
}

void saveImages(const std::vector<cv::Mat>& images,
                const int start_frame,
                const std::string& out_image_pattern) {
  const boost::format out_image_names(out_image_pattern);

#pragma omp parallel for
  for (int i = 0; i < images.size(); ++i) {
    const int frame_id = i + start_frame;
    cv::imwrite((boost::format(out_image_names) % frame_id).str(), images[i]);
  }
}

}  // namespace vp
