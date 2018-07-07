/**
 * @file ColorizeFlow.cpp
 * @brief ColorizeFlow
 *
 * @author Abhijit Kundu
 */

#define _USE_MATH_DEFINES
#include <cmath>
#include "VideoParsing/Utils/ColorizeFlow.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace vp {

cv::Mat colorizeFlow(const cv::Mat& flow) {
  cv::Mat color_flow(flow.size(), CV_32FC3);
  float max_magnitude = std::numeric_limits<float>::lowest();
  for (int r = 0; r < flow.rows; ++r)
    for (int c = 0; c < flow.cols; ++c) {
      cv::Vec2f uv = flow.at<cv::Vec2f>(r, c);
      float magnitude = cv::norm(uv);
      float angle = std::atan2(uv[1], uv[0]) * 180.f / M_PI;
      color_flow.at<cv::Vec3f>(r,c) = cv::Vec3f(angle, 1.f, magnitude);
      max_magnitude = std::max(max_magnitude, magnitude);
    }

  for (int r = 0; r < flow.rows; ++r)
    for (int c = 0; c < flow.cols; ++c) {
      color_flow.at<cv::Vec3f>(r,c)[2] /= max_magnitude;
    }

  cv::cvtColor(color_flow, color_flow, cv::COLOR_HSV2BGR);
  return color_flow;
}

cv::Mat colorizeFlow(const cv::Mat& flow, float max_magnitude) {
  cv::Mat color_flow(flow.size(), CV_32FC3);
  for (int r = 0; r < flow.rows; ++r)
    for (int c = 0; c < flow.cols; ++c) {
      cv::Vec2f uv = flow.at<cv::Vec2f>(r, c);
      float magnitude = cv::norm(uv);
      float angle = std::atan2(uv[1], uv[0]) * 180.f / M_PI;
      color_flow.at<cv::Vec3f>(r,c) = cv::Vec3f(angle, 1.f, magnitude / max_magnitude);
    }
  cv::cvtColor(color_flow, color_flow, cv::COLOR_HSV2BGR);
  return color_flow;
}


cv::Vec3b hsvToBgr8U (float h, float s, float v) {
  float r=0;
  float g=0;
  float b=0;

  float c  = v*s;
  float h2 = 6.0*h;
  float x  = c*(1.0-fabs(fmod(h2,2.0)-1.0));
  if (0<=h2&&h2<1)       { r = c; g = x; b = 0; }
  else if (1<=h2&&h2<2)  { r = x; g = c; b = 0; }
  else if (2<=h2&&h2<3)  { r = 0; g = c; b = x; }
  else if (3<=h2&&h2<4)  { r = 0; g = x; b = c; }
  else if (4<=h2&&h2<5)  { r = x; g = 0; b = c; }
  else if (5<=h2&&h2<=6) { r = c; g = 0; b = x; }
  else if (h2>6) { r = 1; g = 0; b = 0; }
  else if (h2<0) { r = 0; g = 1; b = 0; }

  return cv::Vec3b(255. * b, 255. * g, 255. * r);
}


cv::Mat colorizeFlowKIITI(const cv::Mat& flow, float max_magnitude) {

  if(max_magnitude <= 0) {
    max_magnitude = std::numeric_limits<float>::lowest();

    for (int r = 0; r < flow.rows; ++r)
      for (int c = 0; c < flow.cols; ++c) {
        cv::Vec2f uv = flow.at<cv::Vec2f>(r, c);
        float magnitude = cv::norm(uv);
        max_magnitude = std::max(max_magnitude, magnitude);
      }
    max_magnitude = std::max(max_magnitude, 1.0f);
  }

  cv::Mat color_flow(flow.size(), CV_8UC3);
  for (int r = 0; r < flow.rows; ++r)
    for (int c = 0; c < flow.cols; ++c) {
      cv::Vec2f uv = flow.at<cv::Vec2f>(r, c);
      float magnitude = cv::norm(uv);
      float dir = std::atan2(uv[1], uv[0]);

      const float n = 8; // multiplier

      float h   = std::fmod(dir/(2.0*M_PI)+1.0,1.0);
      float s   = std::min(std::max(magnitude*n/max_magnitude,0.0f),1.0f);
      float v   = std::min(std::max(n-s,0.0f),1.0f);

      color_flow.at<cv::Vec3b>(r,c) = hsvToBgr8U(h,s,v);
    }

  return color_flow;
}

cv::Vec3b getColorFromUV(const float u, const float v, float max_magnitude) {
  float magnitude = std::sqrt(u*u + v*v);
  float dir = std::atan2(v, u);

  const float n = 8;  // multiplier

  float hue = std::fmod(dir / (2.0 * M_PI) + 1.0, 1.0);
  float sat = std::min(std::max(magnitude * n / max_magnitude, 0.0f), 1.0f);
  float val = std::min(std::max(n - sat, 0.0f), 1.0f);
  return hsvToBgr8U(hue, sat, val);
}

cv::Mat getFlowMagnitude(const cv::Mat& flow) {
  assert(flow.type() == CV_32FC2);
  cv::Mat flow_magnitude(flow.size(), CV_32FC1);
  for (int r = 0; r < flow.rows; ++r)
    for (int c = 0; c < flow.cols; ++c) {
      cv::Vec2f uv = flow.at<cv::Vec2f>(r, c);
      flow_magnitude.at<float>(r,c) = cv::norm(uv);
    }
  return flow_magnitude;
}

cv::Mat colorizeFlowWithMagnitude(const cv::Mat& flow, float max_magnitude, int color_map) {
  cv::Mat flow_magnitude = getFlowMagnitude(flow);
  cv::Mat adjMap;
  flow_magnitude.convertTo(adjMap,CV_8UC1, 255 / max_magnitude);
  cv::Mat colored_map;
  cv::applyColorMap(adjMap, colored_map, color_map);
  return colored_map;
}

}  // namespace vp
