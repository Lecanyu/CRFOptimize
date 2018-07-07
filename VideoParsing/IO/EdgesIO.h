/**
 * @file EdgesIO.h
 * @brief EdgesIO
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_IO_EDGESIO_H_
#define VIDEOPARSING_IO_EDGESIO_H_

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

namespace vp {

/**@brief Reads Edge probability map and returns as OpenCV mat
 *
 * The function expects the edge_map to be stored as single channel
 * floating point (32 bit) image like (*.exr).
 *
 * This function can also load from uncompressed binary files (*.bin, *.sed)
 *
 * @param filepath
 * @return single channel floating point (32 bit) image
 */
cv::Mat readEdgeProbAsImage(const std::string& filepath);

/**@brief Reads Edge probability map but returns its as Eigen array.
 *
 * The function expects the edge_map to be stored as single channel
 * floating point (32 bit) image like (*.exr).
 *
 * This function can also load from uncompressed binary files (*.bin, *.sed)
 *
 * @param filepath
 * @return single channel floating point (32 bit) image
 */
Eigen::VectorXf readEdgeProb(const std::string& filepath);

}  // namespace vp


#endif /* VIDEOPARSING_IO_EDGESIO_H_ */
