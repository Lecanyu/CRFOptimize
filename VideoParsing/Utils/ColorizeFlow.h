/**
 * @file ColorizeFlow.h
 * @brief ColorizeFlow
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_UTILS_COLORIZE_FLOW_H_
#define VIDEOPARSING_UTILS_COLORIZE_FLOW_H_

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION == 2
#include <opencv2/contrib/contrib.hpp>
#endif

namespace vp {

cv::Mat colorizeFlow(const cv::Mat& flow);

cv::Mat colorizeFlow(const cv::Mat& flow, float max_magnitude);

cv::Mat colorizeFlowKIITI(const cv::Mat& flow, float max_magnitude = -1.0);

cv::Vec3b getColorFromUV(const float u, const float v, float max_magnitude);

cv::Mat getFlowMagnitude(const cv::Mat& flow);

cv::Mat colorizeFlowWithMagnitude(const cv::Mat& flow, float max_magnitude = 15.0, int color_map = cv::COLORMAP_JET);

}  // namespace vp

#endif /* VIDEOPARSING_UTILS_COLORIZE_FLOW_H_ */
