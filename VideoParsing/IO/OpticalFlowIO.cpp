/**
 * @file OpticalFlowIO.cpp
 * @brief OpticalFlowIO
 *
 * @author Abhijit Kundu
 */

#include "VideoParsing/IO/OpticalFlowIO.h"
#include <fstream>

namespace vp {

const float FLOW_TAG_FLOAT = 202021.25f;
const char *FLOW_TAG_STRING = "PIEH";

cv::Mat readOpticalFlow(const std::string& path) {
  cv::Mat_<cv::Point2f> flow;
  std::ifstream file(path.c_str(), std::ios_base::binary);
  if (!file.good())
    return flow;  // no file - return empty matrix

  float tag;
  file.read((char*) &tag, sizeof(float));
  if (tag != FLOW_TAG_FLOAT)
    return flow;

  int width, height;

  file.read((char*) &width, 4);
  file.read((char*) &height, 4);

  flow.create(height, width);

  for (int i = 0; i < flow.rows; ++i) {
    for (int j = 0; j < flow.cols; ++j) {
      cv::Point2f u;
      file.read((char*) &u.x, sizeof(float));
      file.read((char*) &u.y, sizeof(float));
      if (!file.good()) {
        flow.release();
        return flow;
      }

      flow(i, j) = u;
    }
  }
  file.close();
  return flow;
}

bool writeOpticalFlow(const std::string& path, const cv::Mat& input) {
  const int nChannels = 2;
  if (input.channels() != nChannels || input.depth() != CV_32F
      || path.length() == 0)
    return false;

  std::ofstream file(path.c_str(), std::ofstream::binary);
  if (!file.good())
    return false;

  int nRows, nCols;

  nRows = (int) input.size().height;
  nCols = (int) input.size().width;

  const int headerSize = 12;
  char header[headerSize];
  memcpy(header, FLOW_TAG_STRING, 4);
  // size of ints is known - has been asserted in the current function
  memcpy(header + 4, reinterpret_cast<const char*>(&nCols), sizeof(nCols));
  memcpy(header + 8, reinterpret_cast<const char*>(&nRows), sizeof(nRows));
  file.write(header, headerSize);
  if (!file.good())
    return false;

//    if ( input.isContinuous() ) //matrix is continous - treat it as a single row
//    {
//        nCols *= nRows;
//        nRows = 1;
//    }

  for (int row = 0; row < nRows; ++row) {
    file.write(input.ptr<char>(row), nCols * nChannels * sizeof(float));
    if (!file.good())
      return false;
  }
  file.close();
  return true;
}

}  // namespace vp
