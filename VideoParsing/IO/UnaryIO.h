/**
 * @file UnaryIO.h
 * @brief UnaryIO
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_IO_UNARYIO_H_
#define VIDEOPARSING_IO_UNARYIO_H_

#include <opencv2/core/core.hpp>
#include <Eigen/Core>

namespace vp {

/**@brief Loads unary data
 *
 * @param filepath
 * @return a Eigen matrix with unary data of size num_of_labels X num_of_pixels
 */
Eigen::MatrixXf loadUnary(const std::string& filepath);

/**@brief Loads unary data and also stores the width and height
 *
 * @param[in] filepath
 * @param[out] width
 * @param[out] height
 * @return a Eigen matrix with unary data of size num_of_labels X num_of_pixels
 */
Eigen::MatrixXf loadUnary(const std::string& filepath, int& width, int& height);

/**@brief Loads unary data and returns a OpenCV mat
 *
 * @param filepath
 * @return a OpenCV mat with number of channels = num_of_labels
 */
cv::Mat loadUnaryAsImage(const std::string& filepath);

/**@brief Saves the unary in an uncompresses binary file
 *
 * Output is a binary file with the following format:
 * width(int)
 * height(int)
 * num_of_labels(int)
 * unary_data(floating point array of size W * H * num_of_labels)
 *
 * @param unary input Eigen matrix with unary data of size num_of_labels X num_of_pixels i.e W*H
 * @param W width of the image
 * @param H height of the image
 * @param filepath save file path
 */
void saveUnaryAsUncompressedBinaryFile(Eigen::MatrixXf& unary, const int W, const int H, const std::string& filepath);

}  // namespace vp

#endif /* VIDEOPARSING_IO_UNARYIO_H_ */
