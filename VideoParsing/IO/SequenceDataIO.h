/**
 * @file SequenceDataIO.h
 * @brief SequenceDataIO
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_IO_SEQUENCEDATAIO_H_
#define VIDEOPARSING_IO_SEQUENCEDATAIO_H_

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <vector>

namespace vp {

std::vector<cv::Mat> loadImageSequence(const std::string& file_pattern,
                                       int start_frame,
                                       int number_of_frames);

std::vector<Eigen::MatrixXf> loadUnaries(const std::string& file_pattern,
                                         int start_frame,
                                         int number_of_frames,
                                         bool negate = true);

std::vector<Eigen::MatrixXf> loadUnariesAsNegLogProb(const std::string& file_pattern,
                                                     int start_frame,
                                                     int number_of_frames);

std::vector<Eigen::MatrixXf> loadUnariesAsProb(const std::string& file_pattern,
                                               int start_frame,
                                               int number_of_frames);

std::vector<cv::Mat> loadOpticalFlows(const std::string& file_pattern,
                                      int start_frame,
                                      int number_of_frames);


}  // namespace vp


#endif /* VIDEOPARSING_IO_SEQUENCEDATAIO_H_ */
