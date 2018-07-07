/**
 * @file EigenSerialization.h
 * @brief EigenSerialization
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_CORE_EIGENSERIALIZATION_H_
#define VIDEOPARSING_CORE_EIGENSERIALIZATION_H_

#include <Eigen/Core>
#include <fstream>

namespace vp {

template <class Derived>
void saveEigenMatrix(const Eigen::MatrixBase<Derived>& mat, const std::string& filepath) {

  std::ofstream ofile(filepath, std::ios::binary);

  std::size_t rows = mat.rows();
  std::size_t cols = mat.cols();
  std::size_t salar_size = sizeof(typename Derived::Scalar);

  ofile.write((char*) &rows, sizeof(rows));
  ofile.write((char*) &cols, sizeof(cols));
  ofile.write((char*) &salar_size, sizeof(salar_size));
  ofile.write((char *) mat.derived().data(), salar_size * mat.size());

  ofile.close();
}

template<class Derived>
void loadEigenMatrix(Eigen::MatrixBase<Derived> const& mat, const std::string& filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open File!");
  }
  std::size_t rows, cols, scalar_size;

  file.read((char *) &rows, sizeof(rows));
  file.read((char *) &cols, sizeof(cols));
  file.read((char *) &scalar_size, sizeof(scalar_size));

  if(sizeof(typename Derived::Scalar) != scalar_size) {
    throw std::runtime_error("Scalar Size is Different!");
  }

  Eigen::MatrixBase<Derived>& m = const_cast<Eigen::MatrixBase<Derived>&>(mat);
  m.derived().resize(rows, cols);  // resize the derived object

  file.read((char *) m.derived().data(), sizeof(typename Derived::Scalar) * m.size());

  file.close();

}

}  // namespace vp



#endif /* VIDEOPARSING_CORE_EIGENSERIALIZATION_H_ */
