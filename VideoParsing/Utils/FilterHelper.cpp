/**
 * @file FilterHelper.cpp
 * @brief FilterHelper
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/Utils/FilterHelper.h"
#include "VideoParsing/Core/CoordToIndex.h"
#include <fstream>

namespace vp {

Eigen::Matrix<double, 2, Eigen::Dynamic> createFeaturesXY(int width, int height, int number_of_frames) {
  Eigen::Matrix<double, 2, Eigen::Dynamic> features(2, width * height * number_of_frames);
  Eigen::Index index = 0;
  for (int t = 0; t < number_of_frames; ++t)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {

        features(0, index) = x;
        features(1, index) = y;

        ++index;
      }
  return features;
}


Eigen::MatrixXf createFeaturesXY(float sx, float sy, int width, int height) {
  Eigen::MatrixXf feature(2, width * height);
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      const int index = y * width + x;

      feature(0, index) = x / sx;
      feature(1, index) = y / sy;
    }
  return feature;
}

Eigen::MatrixXf createFeaturesXY(float sx, float sy, const cv::Mat& flow) {
  Eigen::MatrixXf feature(2, flow.rows * flow.cols);
  for (int y = 0; y < flow.rows; y++)
    for (int x = 0; x < flow.cols; x++) {
      const int index = y * flow.cols + x;

      cv::Vec2f uv = flow.at<cv::Vec2f>(y, x);
      // TODO check bound or consistency

      feature(0, index) = (x + uv[0]) / sx;
      feature(1, index) = (y + uv[1]) / sy;
    }
  return feature;
}

Eigen::Matrix<float, 5, Eigen::Dynamic> createFeaturesXYRGB(float sx, float sy, float sr, float sg, float sb, const cv::Mat& color_image) {
  Eigen::Matrix<float, 5, Eigen::Dynamic> feature(5, color_image.rows * color_image.cols);
  for (int y = 0; y < color_image.rows; y++)
    for (int x = 0; x < color_image.cols; x++) {
      const int index = y * color_image.cols + x;

      feature(0, index) = x / sx;
      feature(1, index) = y / sy;

      cv::Vec3b bgr = color_image.at<cv::Vec3b>(y, x);
      feature(2, index) = bgr[2] / sr;
      feature(3, index) = bgr[1] / sg;
      feature(4, index) = bgr[0] / sb;
    }
  return feature;
}

Eigen::Matrix<float, 7, Eigen::Dynamic> createFeaturesXYRGBUV(float sx, float sy, float sr, float sg, float sb, float su, float sv, const cv::Mat& color_image, const cv::Mat& flow_image) {
  Eigen::Matrix<float, 7, Eigen::Dynamic> feature(7, color_image.rows * color_image.cols);
  for (int y = 0; y < color_image.rows; y++)
    for (int x = 0; x < color_image.cols; x++) {
      const int index = y * color_image.cols + x;

      feature(0, index) = x / sx;
      feature(1, index) = y / sy;

      cv::Vec3b bgr = color_image.at<cv::Vec3b>(y, x);
      feature(2, index) = bgr[2] / sr;
      feature(3, index) = bgr[1] / sg;
      feature(4, index) = bgr[0] / sb;

      cv::Vec2f uv = flow_image.at<cv::Vec2f>(y, x);
      feature(5, index) = uv[0] / su;
      feature(6, index) = uv[1] / sv;
    }
  return feature;
}

Eigen::MatrixXf createFeaturesXYRGB(float sx, float sy, float sr, float sg, float sb, const cv::Mat& color_image, const cv::Mat& flow) {
  Eigen::MatrixXf feature(5, color_image.rows * color_image.cols);
  for (int y = 0; y < color_image.rows; y++)
    for (int x = 0; x < color_image.cols; x++) {
      const int index = y * color_image.cols + x;

      cv::Vec2f uv = flow.at<cv::Vec2f>(y, x);
      // TODO check for consistency

      feature(0, index) = (x + uv[0]) / sx;
      feature(1, index) = (y + uv[1]) / sy;

      cv::Vec3b bgr = color_image.at<cv::Vec3b>(y, x);
      feature(2, index) = bgr[2] / sr;
      feature(3, index) = bgr[1] / sg;
      feature(4, index) = bgr[0] / sb;
    }
  return feature;
}

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesRGB(float sr, float sg, float sb, const cv::Mat& color_image) {
  Eigen::Matrix<float, 3, Eigen::Dynamic> feature(3, color_image.rows * color_image.cols);
  for (int y = 0; y < color_image.rows; y++)
    for (int x = 0; x < color_image.cols; x++) {
      const int index = y * color_image.cols + x;
      cv::Vec3b bgr = color_image.at<cv::Vec3b>(y, x);
      feature(0, index) = bgr[2] / sr;
      feature(1, index) = bgr[1] / sg;
      feature(2, index) = bgr[0] / sb;
    }
  return feature;
}

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesXYS(float sx, float sy, float std_dev, const cv::Mat& seg) {
  Eigen::Matrix<float, 3, Eigen::Dynamic> feature(3, seg.rows * seg.cols);
  for (int y = 0; y < seg.rows; y++)
    for (int x = 0; x < seg.cols; x++) {
      const int index = y * seg.cols + x;
      feature(0, index) = x / sx;
      feature(1, index) = y / sy;
      feature(2, index) = seg.at<int>(y, x) / std_dev;
    }
  return feature;
}

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesXYS(float sx, float sy, const cv::Mat& flow, float std_dev, const cv::Mat& seg) {
  Eigen::Matrix<float, 3, Eigen::Dynamic> feature(3, seg.rows * seg.cols);
  for (int y = 0; y < seg.rows; y++)
    for (int x = 0; x < seg.cols; x++) {
      const int index = y * seg.cols + x;

      cv::Vec2f uv = flow.at<cv::Vec2f>(y, x);
      feature(0, index) = (x + uv[0]) / sx;
      feature(1, index) = (y + uv[1]) / sy;
      feature(2, index) = seg.at<int>(y, x) / std_dev;
    }
  return feature;
}

Eigen::MatrixXf createFeaturesS(float std_dev, const cv::Mat& seg) {
  Eigen::MatrixXf feature(1, seg.rows * seg.cols);
  for (int y = 0; y < seg.rows; y++)
    for (int x = 0; x < seg.cols; x++) {
      const int index = y * seg.cols + x;
      feature(0, index) = seg.at<int>(y, x) / std_dev;
    }
  return feature;
}

std::vector<Eigen::MatrixXf> createFeaturesXY(float sx, float sy, const std::vector<cv::Mat>& fwd_flows) {
  std::size_t number_of_frames = fwd_flows.size() + 1;

  std::vector<Eigen::MatrixXf> features(number_of_frames);

  const int width = fwd_flows.front().cols;
  const int height = fwd_flows.front().rows;

  {
    Eigen::MatrixXf featureXY(2, width * height);

    int index = 0;
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {

        featureXY(0, index) = x / sx;
        featureXY(1, index) = y / sy;

        ++index;
      }
    features[number_of_frames - 1] = featureXY;
  }

  for (int i = number_of_frames - 2; i >= 0; --i) {
    features[i] = features[i+1];

    int index = 0;
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        cv::Vec2f uv = fwd_flows[i].at<cv::Vec2f>(y, x);

        features[i](0, index) += uv[0] / sx;
        features[i](1, index) += uv[1] / sy;

        ++index;
      }
  }

  return features;
}


std::vector<Eigen::MatrixXf> createFeaturesXYRGB(
    float sx, float sy, float sr, float sg, float sb,
    const std::vector<cv::Mat>& flows,
    const std::vector<cv::Mat>& images) {

  if(flows.size() != (images.size() - 1))
    throw std::length_error("# of images != (# of flows + 1)");

  std::size_t number_of_frames = images.size();

  const int width = flows.front().cols;
  const int height = flows.front().rows;

  // Initialize the vector of features
  std::vector<Eigen::MatrixXf> features(number_of_frames, Eigen::MatrixXf(5, width * height));

  {
    const int frame_id = number_of_frames - 1;

    int index = 0;
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {

        features[frame_id](0, index) = x / sx;
        features[frame_id](1, index) = y / sy;

        cv::Vec3b bgr = images[frame_id].at<cv::Vec3b>(y, x);
        features[frame_id](2, index) = bgr[2] / sr;
        features[frame_id](3, index) = bgr[1] / sg;
        features[frame_id](4, index) = bgr[0] / sb;

        ++index;
      }
  }

  for (int frame_id = number_of_frames - 2; frame_id >= 0; --frame_id) {
    int index = 0;
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        cv::Vec2f uv = flows[frame_id].at<cv::Vec2f>(y, x);

        features[frame_id](0, index) = features[frame_id+1](0, index) + uv[0] / sx;
        features[frame_id](1, index) = features[frame_id+1](1, index) + uv[1] / sy;

        cv::Vec3b bgr = images[frame_id].at<cv::Vec3b>(y, x);
        features[frame_id](2, index) = bgr[2] / sr;
        features[frame_id](3, index) = bgr[1] / sg;
        features[frame_id](4, index) = bgr[0] / sb;

        ++index;
      }
  }

  return features;
}


Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesXYT(float sx, float sy, float st,
                                                          const std::vector<cv::Mat>& flows) {
  int num_of_frames = flows.size() + 1;
  int width = flows.front().cols;
  int height = flows.front().rows;

  typedef CoordToIndex<3, Eigen::DenseIndex> CoordToIndex3;
  const CoordToIndex3 coordToindex(width, height, num_of_frames);

  Eigen::Matrix<float, 3, Eigen::Dynamic>  feature(3, coordToindex.maxIndex());

  const int last_frame_id = num_of_frames - 1;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      const Eigen::DenseIndex index = coordToindex(x, y, last_frame_id);

      feature(0, index) = x / sx;
      feature(1, index) = y / sy;
      feature(2, index) = last_frame_id / st;
    }

  for (int t = last_frame_id - 1; t >= 0; --t)
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++) {
        const Eigen::DenseIndex curr_index = coordToindex(x, y, t);
        const Eigen::DenseIndex next_img_index = coordToindex(x, y, t + 1);

        const cv::Vec2f uv = flows[t].at<cv::Vec2f>(y, x);
        feature(0, curr_index) = uv[0] / sx + feature(0, next_img_index);
        feature(1, curr_index) = uv[1] / sy + feature(1, next_img_index);
        feature(2, curr_index) = t / st;
      }
  return feature;
}

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesXYT(float sx, float sy, float st, int W, int H, int T) {

  typedef CoordToIndex<3, Eigen::DenseIndex> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, T);

  Eigen::Matrix<float, 3, Eigen::Dynamic>  feature(3, coordToindex.maxIndex());

  for (int t = 0; t < T; ++t)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        const Eigen::DenseIndex curr_index = coordToindex(x, y, t);
        feature(0, curr_index) = x / sx;
        feature(1, curr_index) = y / sy;
        feature(2, curr_index) = t / st;
      }
  return feature;
}

Eigen::Matrix<float, 6, Eigen::Dynamic> createFeaturesXYTRGB(float sx, float sy, float st,
                                                             float sr, float sg, float sb,
                                                             const std::vector<cv::Mat>& flows,
                                                             const std::vector<cv::Mat>& images) {
  if (flows.size() != (images.size() - 1))
    throw std::length_error("# of images != (# of flows + 1)");

  int width = flows.front().cols;
  int height = flows.front().rows;

  typedef CoordToIndex<3, Eigen::DenseIndex> CoordToIndex3;
  const CoordToIndex3 coordToindex(width, height, images.size());

  Eigen::Matrix<float, 6, Eigen::Dynamic>  feature(6, coordToindex.maxIndex());

  const int last_frame_id = images.size() - 1;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      const Eigen::DenseIndex index = coordToindex(x, y, last_frame_id);

      feature(0, index) = x / sx;
      feature(1, index) = y / sy;
      feature(2, index) = last_frame_id / st;

      cv::Vec3b bgr = images[last_frame_id].at<cv::Vec3b>(y, x);
      feature(3, index) = bgr[2] / sr;
      feature(4, index) = bgr[1] / sg;
      feature(5, index) = bgr[0] / sb;
    }

  for (int t = last_frame_id - 1; t >= 0; --t)
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++) {
        const Eigen::DenseIndex curr_index = coordToindex(x, y, t);
        const Eigen::DenseIndex next_img_index = coordToindex(x, y, t + 1);

        cv::Vec2f uv = flows[t].at<cv::Vec2f>(y, x);
        feature(0, curr_index) = uv[0] / sx + feature(0, next_img_index);
        feature(1, curr_index) = uv[1] / sy + feature(1, next_img_index);
        feature(2, curr_index) = t / st;

        cv::Vec3b bgr = images[t].at<cv::Vec3b>(y, x);
        feature(3, curr_index) = bgr[2] / sr;
        feature(4, curr_index) = bgr[1] / sg;
        feature(5, curr_index) = bgr[0] / sb;
      }
  return feature;
}

Eigen::Matrix<float, 6, Eigen::Dynamic> createFeaturesXYTRGB(float sx, float sy, float st,
                                                             float sr, float sg, float sb,
                                                             const std::vector<cv::Mat>& images) {
  int W = images.front().cols;
  int H = images.front().rows;
  int T = images.size();

  typedef CoordToIndex<3, Eigen::DenseIndex> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, T);

  Eigen::Matrix<float, 6, Eigen::Dynamic>  feature(6, coordToindex.maxIndex());

  for (int t = 0; t < T; ++t)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
        const Eigen::DenseIndex curr_index = coordToindex(x, y, t);

        feature(0, curr_index) = x / sx;
        feature(1, curr_index) = y / sy;
        feature(2, curr_index) = t / st;

        cv::Vec3b bgr = images[t].at<cv::Vec3b>(y, x);
        feature(3, curr_index) = bgr[2] / sr;
        feature(4, curr_index) = bgr[1] / sg;
        feature(5, curr_index) = bgr[0] / sb;
      }
  return feature;
}

Eigen::Matrix<float, 3, Eigen::Dynamic> createFeaturesRGB(float sr, float sg, float sb, const std::vector<cv::Mat>& images) {
  int W = images.front().cols;
  int H = images.front().rows;
  int T = images.size();

  typedef CoordToIndex<3, Eigen::DenseIndex> CoordToIndex3;
  const CoordToIndex3 coordToindex(W, H, T);

  Eigen::Matrix<float, 3, Eigen::Dynamic>  feature(3, coordToindex.maxIndex());

  for (int t = 0; t < T; ++t)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
        const Eigen::DenseIndex curr_index = coordToindex(x, y, t);
        cv::Vec3b bgr = images.at(t).at<cv::Vec3b>(y, x);
        feature(0, curr_index) = bgr[2] / sr;
        feature(1, curr_index) = bgr[1] / sg;
        feature(2, curr_index) = bgr[0] / sb;
      }
  return feature;
}


}  // namespace vp


