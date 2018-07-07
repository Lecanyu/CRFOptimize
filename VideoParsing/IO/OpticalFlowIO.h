/**
 * @file OpticalFlowIO.h
 * @brief OpticalFlowIO
 *
 * @author Abhijit Kundu
 */
#ifndef VIDEOPARSING_IO_OPTICALFLOWIO_H_
#define VIDEOPARSING_IO_OPTICALFLOWIO_H_

#include <opencv2/core/core.hpp>

namespace vp {

/**@brief Read a .flo file
 *
 * @param Path to the .flo file to be loaded
 *
 * The function readOpticalFlow loads a flow field from a file
 * and returns it as a single matrix.Resulting Mat has
 * a type CV_32FC2 - floating-point, 2-channel. First channel
 * corresponds to the flow in the horizontal direction (u),
 * second - vertical (v).
 *
 */
cv::Mat readOpticalFlow(const std::string& path );

/**@brief Write to a .flo file
 *
 * @param path Path to the file to be written
 * @param flow Flow field to be stored
 *
 * The function stores a flow field in a file, returns true on success, false otherwise.
 * The flow field must be a 2-channel, floating-point matrix (CV_32FC2). First channel corresponds
 * to the flow in the horizontal direction (u), second - vertical (v).
 *
 */
bool writeOpticalFlow(const std::string& path, const cv::Mat& flow );

}  // namespace vp

#endif /* VIDEOPARSING_IO_OPTICALFLOWIO_H_ */
