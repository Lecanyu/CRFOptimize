/**
 * @file SequenceDataIO.cpp
 * @brief SequenceDataIO
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/SequenceDataIO.h"
#include "VideoParsing/IO/UnaryIO.h"
#include "VideoParsing/IO/OpticalFlowIO.h"
#include "VideoParsing/Core/EigenUtils.h"

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace vp {

std::vector<cv::Mat> loadImageSequence(const std::string& file_pattern, int start_frame,
                                       int number_of_frames) {

  const boost::format image_files(file_pattern);
  std::vector<cv::Mat> images(number_of_frames);

#pragma omp parallel for
  for (int i = 0; i < number_of_frames; ++i) {
    const int frame_id = start_frame + i;
    const std::string image_file = (boost::format(image_files) % frame_id).str();

    if (!boost::filesystem::exists(image_file)) {
      std::string error =  image_file + " does not exist";
      throw std::runtime_error(error);
    }

    images[i] = cv::imread(image_file);
  }

  return images;
}


std::vector<Eigen::MatrixXf> loadUnaries(const std::string& file_pattern,
                                         int start_frame,
                                         int number_of_frames,
                                         bool negate) {

  const boost::format unary_files(file_pattern);
  std::vector<Eigen::MatrixXf> unaries(number_of_frames);

#pragma omp parallel for
  for (int i = 0; i < number_of_frames; ++i) {
    const int frame_id = start_frame + i;
    const std::string unary_file = (boost::format(unary_files) % frame_id).str();

    if (!boost::filesystem::exists(unary_file)) {
      std::string error =  unary_file + " does not exist";
      throw std::runtime_error(error);
    }

    if(negate)
      unaries[i] = - loadUnary(unary_file);
    else
      unaries[i] = loadUnary(unary_file);
  }

  return unaries;
}

std::vector<Eigen::MatrixXf> loadUnariesAsNegLogProb(const std::string& file_pattern,
                                                     int start_frame, int number_of_frames) {
  const boost::format unary_files(file_pattern);
  std::vector<Eigen::MatrixXf> unaries(number_of_frames);

#pragma omp parallel for
  for (int i = 0; i < number_of_frames; ++i) {
    const int frame_id = start_frame + i;
    const std::string unary_file = (boost::format(unary_files) % frame_id).str();

    if (!boost::filesystem::exists(unary_file)) {
      std::string error = unary_file + " does not exist";
      throw std::runtime_error(error);
    }

    unaries[i] = -loadUnary(unary_file).array().log();
  }
  return unaries;
}

std::vector<Eigen::MatrixXf> loadUnariesAsProb(const std::string& file_pattern,
                                               int start_frame,
                                               int number_of_frames) {
  const boost::format unary_files(file_pattern);
  std::vector<Eigen::MatrixXf> unaries(number_of_frames);

#pragma omp parallel for
  for (int i = 0; i < number_of_frames; ++i) {
    const int frame_id = start_frame + i;
    const std::string unary_file = (boost::format(unary_files) % frame_id).str();

    if (!boost::filesystem::exists(unary_file)) {
      std::string error = unary_file + " does not exist";
      throw std::runtime_error(error);
    }

    unaries[i] = loadUnary(unary_file);
    assert(unaries[i].colwise().sum().isApproxToConstant(1.0f));
  }
  return unaries;

}


std::vector<cv::Mat> loadOpticalFlows(const std::string& file_pattern,
                                      int start_frame,
                                      int number_of_frames) {
  const boost::format files(file_pattern);
  std::vector<cv::Mat> out(number_of_frames);

#pragma omp parallel for
  for (int i = 0; i < number_of_frames; ++i) {
    const int frame_id = start_frame + i;
    const std::string file = (boost::format(files) % frame_id).str();

    if (!boost::filesystem::exists(file)) {
      std::string error =  file + " does not exist";
      throw std::runtime_error(error);
    }

    out[i] = readOpticalFlow(file);
  }

  return out;
}

}  // namespace vp


