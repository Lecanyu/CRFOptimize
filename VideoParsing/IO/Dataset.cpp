/**
 * @file Dataset.cpp
 * @brief Dataset
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/Dataset.h"
#include "VideoParsing/IO/SequenceDataIO.h"
#include "VideoParsing/IO/UnaryIO.h"
#include "VideoParsing/IO/OpticalFlowIO.h"
#include "VideoParsing/IO/TracksIO.h"
#include "VideoParsing/IO/EdgesIO.h"

#include "VideoParsing/Core/Config.h"
#include "VideoParsing/Core/EigenUtils.h"
#include "VideoParsing/Core/EigenSerialization.h"

#include <iostream>

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/highgui/highgui.hpp>

namespace vp {

namespace detail {

inline std::string getFilePath(const std::string& files, const int frame_id) {
  return (boost::format(files) % frame_id).str();
}

inline void prefixRootPath(std::string& filename, const boost::filesystem::path& root_dir_path) {
  filename = (root_dir_path / boost::filesystem::path(filename)).string();
}

}  // namespace detail

Dataset::Dataset(const std::string& config_filename) {
  namespace fs = boost::filesystem;
  namespace po = boost::program_options;

  fs::path config_fp(config_filename);

  if (!fs::exists(config_fp)) {
    std::cout << "Provided ConfigFile path does not exist: " << config_fp << "\n";
    throw std::runtime_error("Config File Not Found");
  }

  if (!fs::is_regular_file(config_fp)) {
    std::cout << "Provided ConfigFile is not a regular file: " << config_fp << "\n";
    throw std::runtime_error("Config File is not a regular file");
  }

  std::ifstream ifs(config_fp.c_str());
  if (!ifs) {
    std::cout << "failed to open config file from " << config_fp << "\n";
    throw std::runtime_error("Config File Cannot be loaded");
  }

  po::options_description config_options("Dataset");
  config_options.add_options()
      ("name", po::value<std::string>(&name_)->required(), "Dataset Name")
      ("root_dir", po::value<std::string>(&root_dir_)->required(), "Dataset Root directory")
      ("image_width", po::value<int>(&image_width_)->required(), "Image Width")
      ("image_height", po::value<int>(&image_height_)->required(), "Image Height")
      ("rgb_files", po::value<std::string>(&rgb_image_files_wc_), "RGB Image File Wildcard")
      ("unary_files", po::value<std::string>(&unary_files_wc_), "Unary Image Files Wildcard")
      ("fwd_flow_files", po::value<std::string>(&fwd_flow_files_wc_), "Fwd Optical Flow Files Wildcard")
      ("bwd_flow_files", po::value<std::string>(&bwd_flow_files_wc_), "Bwd Optical Flow Files Wildcard")
      ("edge_prob_files", po::value<std::string>(&edge_prob_files_wc_), "Edge Probability Files Wildcard")
      ("tracks_file", po::value<std::string>(&tracks_file_), "Tracks Filepath")
      ;

  po::variables_map vm;
  po::store(po::parse_config_file(ifs, config_options), vm);
  po::notify(vm);

  ifs.close();

  fs::path root_dir_path(root_dir_);

  if (!fs::exists(root_dir_path)) {
    std::cout << "Provided root_dir path does not exist: " << root_dir_path << "\n";
    throw std::runtime_error("Dataset root_dir no found");
  }

  if (!fs::is_directory(root_dir_path)) {
    std::cout << "Provided root_dir is not a directory: " << root_dir_path << "\n";
    throw std::runtime_error("Dataset root_dir is not a directory");
  }

  {
    detail::prefixRootPath(rgb_image_files_wc_, root_dir_path);
    detail::prefixRootPath(unary_files_wc_, root_dir_path);
    detail::prefixRootPath(fwd_flow_files_wc_, root_dir_path);
    detail::prefixRootPath(bwd_flow_files_wc_, root_dir_path);
    detail::prefixRootPath(edge_prob_files_wc_, root_dir_path);
    detail::prefixRootPath(tracks_file_, root_dir_path);
  }

}

void Dataset::print() const {
  namespace bfs = boost::filesystem;
  std::cout << "Dataset Name: " <<  name_ << " (" << image_width_ << "x" << image_height_ << ")\n";
  std::cout << "Unaries: " <<  bfs::path(unary_files_wc_).parent_path().stem() << "   ";
  std::cout << "Fwd/Bwd Flow: " << bfs::path(fwd_flow_files_wc_).parent_path().stem() << " / " <<  bfs::path(bwd_flow_files_wc_).parent_path().stem() << "  ";
  std::cout << "Tracks: " <<  bfs::path(tracks_file_).stem() << "  ";
  std::cout << "EdgeProb: " <<  bfs::path(edge_prob_files_wc_).parent_path().stem() << "\n";
}

void Dataset::printFull() const {
  std::cout << "Dataset Name: " <<  name_ << "\n";
  std::cout << "Image Size: " <<  image_width_ << "x" << image_height_ << "\n";
  std::cout << "RGB files: " <<  rgb_image_files_wc_ << "\n";
  std::cout << "Unary files: " <<  unary_files_wc_ << "\n";
  std::cout << "Fwd Flow files: " <<  fwd_flow_files_wc_ << "\n";
  std::cout << "Bwd Flow files: " <<  bwd_flow_files_wc_ << "\n";
  std::cout << "Edge Prob Files: " <<  edge_prob_files_wc_ << "\n";
  std::cout << "Tracks Filepath: " <<  tracks_file_ << "\n";
}

std::vector<Eigen::MatrixXf> Dataset::loadUnariesAsNegLogProb(int start_frame,
                                                              int number_of_frames) const {
  return ::vp::loadUnariesAsNegLogProb(unary_files_wc_, start_frame, number_of_frames);
}

std::vector<Eigen::MatrixXf> Dataset::loadUnariesAsProb(int start_frame, int number_of_frames) const {
  return ::vp::loadUnariesAsProb(unary_files_wc_, start_frame, number_of_frames);
}

std::vector<Eigen::MatrixXf> Dataset::loadUnaries(int start_frame, int number_of_frames) const {
  return ::vp::loadUnaries(unary_files_wc_, start_frame, number_of_frames, false);
}

std::vector<cv::Mat> Dataset::loadImageSequence(int start_frame, int number_of_frames) const {
  return ::vp::loadImageSequence(rgb_image_files_wc_, start_frame, number_of_frames);
}

std::vector<cv::Mat> Dataset::loadForwardFlows(int start_frame, int number_of_frames) const {
  return loadOpticalFlows(fwd_flow_files_wc_, start_frame, number_of_frames);
}

std::vector<cv::Mat> Dataset::loadBackwardFlows(int start_frame, int number_of_frames) const {
  return loadOpticalFlows(bwd_flow_files_wc_, start_frame, number_of_frames);
}

cv::Mat Dataset::loadImage(int frame_id) const {
  return cv::imread(detail::getFilePath(rgb_image_files_wc_, frame_id), CV_LOAD_IMAGE_COLOR);
}

cv::Mat Dataset::loadFwdOpticalFlow(int frame_id) const {
  return readOpticalFlow(detail::getFilePath(fwd_flow_files_wc_, frame_id));
}

cv::Mat Dataset::loadBwdOpticalFlow(int frame_id) const {
  return readOpticalFlow(detail::getFilePath(bwd_flow_files_wc_, frame_id));
}

Eigen::MatrixXf Dataset::loadUnaryAsNegLogProb(int frame_id) const {
 return -(loadUnary(detail::getFilePath(unary_files_wc_, frame_id))).array().log();
}

Eigen::MatrixXf Dataset::loadUnaryAsProb(int frame_id) const {
  return loadUnary(detail::getFilePath(unary_files_wc_, frame_id));
}


cv::Mat Dataset::loadEdgeProbAsImage(int frame_id) const {
  return readEdgeProbAsImage(detail::getFilePath(edge_prob_files_wc_, frame_id));
}

Eigen::VectorXf Dataset::loadEdgeProb(int frame_id) const {
  return readEdgeProb(detail::getFilePath(edge_prob_files_wc_, frame_id));
}

std::vector<Track2D> Dataset::loadTracks() const {
  namespace bf = boost::filesystem;
  bf::path track_fp(tracks_file_);
  if (!bf::exists(track_fp)) {
    std::cout << "Tracks File: "<< track_fp << " does not exist\n";
    throw std::runtime_error("Tracks File does not exist");
  }
 return readTracks(track_fp.string());
}

std::vector<Track2D> Dataset::loadTracks(int start_frame, int number_of_frames) const {
  std::vector<Track2D> all_tracks = loadTracks();
  std::vector<Track2D> tracks;
  std::copy_if(all_tracks.begin(), all_tracks.end(), std::back_inserter(tracks),
               IsValidTrack(start_frame, number_of_frames));
  return tracks;
}

std::string Dataset::imageFileNameStem(int frame_id) const {
  namespace fs = boost::filesystem;
  return fs::path(detail::getFilePath(rgb_image_files_wc_, frame_id)).stem().string();
}


}  // end namespace vp
