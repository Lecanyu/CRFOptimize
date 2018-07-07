/**
 * @file Dataset.h
 * @brief Dataset
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_IO_DATASET_H_
#define VIDEOPARSING_IO_DATASET_H_

#include "VideoParsing/Core/Track2D.h"
#include <opencv2/core/core.hpp>
#include <vector>

namespace vp {

/**@brief Dataset config class
 *
 * Holds several datset related configurations.
 * Dataset class is used to read a dataset config file.
 *
 */
class Dataset {
public:
  /**@brief Dataset constructor from config file
   *
   * @param config_filepath
   */
  Dataset(const std::string& config_filepath);

  void print() const;
  void printFull() const;
  const std::string& name() const {return name_;}
  int imageWidth() const {return image_width_;}
  int imageHeight() const {return image_height_;}

  std::vector<Eigen::MatrixXf> loadUnariesAsNegLogProb(int start_frame, int number_of_frames) const;
  std::vector<Eigen::MatrixXf> loadUnariesAsProb(int start_frame, int number_of_frames) const;
  std::vector<Eigen::MatrixXf> loadUnaries(int start_frame, int number_of_frames) const;

  std::vector<cv::Mat> loadImageSequence(int start_frame, int number_of_frames) const;
  std::vector<cv::Mat> loadForwardFlows(int start_frame, int number_of_frames) const;
  std::vector<cv::Mat> loadBackwardFlows(int start_frame, int number_of_frames) const;

  cv::Mat loadImage(int frame_id) const;
  cv::Mat loadFwdOpticalFlow(int frame_id) const;
  cv::Mat loadBwdOpticalFlow(int frame_id) const;
  Eigen::MatrixXf loadUnaryAsNegLogProb(int frame_id) const;
  Eigen::MatrixXf loadUnaryAsProb(int frame_id) const;

  cv::Mat loadEdgeProbAsImage(int frame_id) const;
  Eigen::VectorXf loadEdgeProb(int frame_id) const;

  std::vector<Track2D> loadTracks() const;
  std::vector<Track2D> loadTracks(int start_frame, int number_of_frames) const;

  /// @returns the image filename (without extension) at frame_id
  std::string imageFileNameStem(int frame_id) const;

 private:
  std::string name_;
  std::string root_dir_;
  int image_width_, image_height_;
  std::string rgb_image_files_wc_;
  std::string unary_files_wc_;
  std::string fwd_flow_files_wc_;
  std::string bwd_flow_files_wc_;
  std::string edge_prob_files_wc_;
  std::string tracks_file_;
};

}  // end namespace vp

#endif /* VIDEOPARSING_IO_DATASET_H_ */
