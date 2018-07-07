/**
 * @file PermutohedralLattice.h
 * @brief PermutohedralLattice
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_DENSEGM_PERMUTOHEDRALLATTICE_H_
#define VIDEOPARSING_DENSEGM_PERMUTOHEDRALLATTICE_H_

#include "VideoParsing/DenseGM/HashTable.h"
#include <Eigen/Core>

namespace vp {

template <int D>
class PermutohedralLattice {
  struct Neighbors {
    int n1, n2;
    Neighbors(int n1 = 0, int n2 = 0)
        : n1(n1),
          n2(n2) {
    }
  };

 public:
  int N_;  //! Number of elements
  int M_;  //! size of sparse discretized space

  Eigen::Matrix<int, D + 1, Eigen::Dynamic> offset_;
  Eigen::Matrix<float, D + 1, Eigen::Dynamic> barycentric_;
  std::vector<Neighbors> blur_neighbors_;



  template <class Derived>
  PermutohedralLattice(const Eigen::MatrixBase<Derived>& feature, std::size_t num_of_lattice_points = 0)
    : N_(feature.cols()),
      offset_(D + 1 , N_),
      barycentric_(D + 1 , N_) {

    static_assert(Derived::RowsAtCompileTime == D, "Feature Dimension is incorrect");

    typedef Eigen::Matrix<float, D, 1> VectorDf;
    typedef Eigen::Matrix<float, D + 1, 1> VectorD1f;
    typedef Eigen::Matrix<float, D + 2, 1> VectorD2f;

    typedef Eigen::Matrix<short, D, 1> VectorDs;
    typedef Eigen::Matrix<short, D + 1, 1> VectorD1s;
    typedef Eigen::Matrix<short, D + 1, D + 1> MatrixD1s;

    // Create a hash table for lattice points.
    // Number of lattice points is bounded by N_ * (D + 1)
    // but usually it is much less than N_

    // If no hint provided (set it to N_)
    if(num_of_lattice_points == 0)
      num_of_lattice_points = N_;

    HashTable<VectorDs> hash_table(num_of_lattice_points);

    // Compute the lattice coordinates for each feature

    // compute the coordinates of the canonical simplex, in which
    // the difference between a contained point and the zero
    // remainder vertex is always in ascending order.
    MatrixD1s canonical;
    for (int j = 0; j < canonical.cols(); ++j) {

      for(int i = 0; i < canonical.rows() - j; ++i)
        canonical(i, j) = j;

      for(int i = canonical.cols() - j; i < canonical.rows(); ++i)
        canonical(i, j) = j - canonical.cols();
    }


    // Compute part of the rotation matrix E that elevates a
    // position vector into the hyperplane

    // Compute the diagonal part of E
    VectorDf scale_factor;
    for (int i = 0; i < D; ++i)
      scale_factor[i] = 1.0f / std::sqrt(float((i + 2) * (i + 1)));

    // Scale by inv Expected standard deviation of our filter
    scale_factor *=  std::sqrt(2.0f / 3.0f) * (D + 1);


    // Primary loop over each feature/position vector
    // Loop over each feature and compute the simplex each feature lies in
    for (int k = 0; k < N_; k++) {


      // First elevate position into the (d+1)-dimensional hyperplane
      VectorD1f elevated;
      {
        const VectorDf cf = scale_factor.cwiseProduct(feature.col(k));
        float sm = 0; // sm contains the sum of 1..n of our faeture vector
        for (int j = D; j > 0; j--) {
          elevated[j] = sm - j * cf[j-1];
          sm += cf[j-1];
        }
        elevated[0] = sm;
      }

      // greedily search for the closest remainder zero lattice point

      // METHOD1
//      VectorD1f rem0;
//      int sum = 0;
//      {
//        const float down_factor = 1.0f / (D + 1);
//        const float up_factor = (D + 1);
//
//        for (int i = 0; i <= D; ++i) {
//          int rd2;
//          float v = down_factor * elevated[i];
//          float up = ceilf(v) * up_factor;
//          float down = floorf(v) * up_factor;
//          if (up - elevated[i] < elevated[i] - down)
//            rd2 = (short) up;
//          else
//            rd2 = (short) down;
//
//          rem0[i] = rd2;
//          sum += rd2 * down_factor;
//        }
//      }


      // METHOD2

      VectorD1f v = elevated / (D+1);
      v = v.array().round();
      VectorD1f rem0 = v *(D +1);
      int sum = v.sum();

      ///-----------------------------------------------------///


      // Find the simplex we are in and store it in rank
      // (where rank describes what position coorinate i has in
      // the sorted order of the features values)
      // (See pg. 3-4 in [Adams et al 2010].)
      VectorD1s rank;
      rank.setZero();

      VectorD1f diff = elevated - rem0;
      for (int i = 0; i < D; ++i) {
        for (int j = i + 1; j <= D; ++j)
          if (diff[i] < diff[j])
            ++rank[i];
          else
            ++rank[j];
      }

      // If the point doesn't lie on the plane (sum != 0) bring it back
      rank.array() += sum;
      for (int i = 0; i <= D; ++i) {
        if (rank[i] < 0) {
          rank[i] += D + 1;
          rem0[i] += D + 1;
        } else if (rank[i] > D) {
          rank[i] -= D + 1;
          rem0[i] -= D + 1;
        }
      }

      // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
      VectorD2f barycentric;
      barycentric.setZero();

      diff = (elevated - rem0) / (D + 1);
      for (int i = 0; i <= D; ++i) {
        barycentric[D - rank[i]] += diff[i];
        barycentric[D - rank[i] + 1] -= diff[i];
      }
      // Wrap around
      barycentric[0] += 1.0f + barycentric[D + 1];

      // Compute all vertices and their offset
      for (int remainder = 0; remainder <= D; remainder++) {
        VectorDs key;
        for (int i = 0; i < D; ++i)
          key[i] = rem0[i] + canonical(rank[i], remainder);

        offset_(remainder, k) = hash_table.insert( key) + 1;
      }
      barycentric_.col(k) = barycentric.template head<D+1>();

    } // end Primary loop over each feature/position vector


    // Get the number of vertices in the lattice
    M_ = hash_table.size();

    // Create the neighborhood structure
    blur_neighbors_.resize((D + 1) * M_);

    // For each of d+1 axes,
    for (int j = 0; j <= D; ++j) {
      // Find the Neighbors of each lattice point
      for( int i=0; i<M_; ++i ) {

        const VectorDs& key = hash_table.getKey(i);

        VectorDs n1 = key.array() - 1;
        VectorDs n2 = key.array() + 1;

        n1[j] = key[j] + D;
        n2[j] = key[j] - D;

        blur_neighbors_[j * M_ + i].n1 = hash_table.find(n1) + 1;
        blur_neighbors_[j * M_ + i].n2 = hash_table.find(n2) + 1;
      }
    }
  }

  template<class Derived>
  typename Derived::PlainObject compute(const Eigen::MatrixBase<Derived>& in, bool reverse = false) const {
    typename Derived::PlainObject out(in.rows(), in.cols());
    compute(out, in, reverse);
    return out;
  }

  template<class OutDerived, class InDerived>
  void compute(Eigen::MatrixBase<OutDerived> const & out, const Eigen::MatrixBase<InDerived>& in, bool reverse = false) const {

    assert(in.cols() == N_);
    assert(in.cols() == out.cols() && in.rows() == out.rows());

    const int value_size = in.rows();

    typename InDerived::PlainObject values(value_size, (M_ + 2));
    values.setZero();

    // Splatting
    for (int j = 0; j < N_; ++j) {
      for (int i = 0; i <= D; ++i) {
        int o = offset_(i, j);
        float w = barycentric_(i, j);

        // Surprisingly this is slow than manual loop on clang
//        values.col(o) += w * in.col(j);

        for (int k = 0; k < value_size; k++)
          values(k, o) += w * in(k, j);
      }
    }

    typename InDerived::PlainObject new_values(value_size, (M_ + 2));
    new_values.setZero();

    // Blurring
    for (int j = reverse ? D : 0; j <= D && j >= 0; reverse ? j-- : ++j) {
      for (int i = 0; i < M_; ++i) {

        const Neighbors& nbr = blur_neighbors_[j * M_ + i];

        new_values.col(i+1) = values.col(i+1) + 0.5 * (values.col(nbr.n1) + values.col(nbr.n2));
      }
      values.swap(new_values);
    }

    // Some magic scaling constant
    const float alpha = 1.0f / (1 + std::pow(2, -D));

    // Start by initializing out to zero
    const_cast< Eigen::MatrixBase<OutDerived>& >(out).setZero();

    // Slicing
    for (int j = 0; j < N_; ++j) {
      for (int i = 0; i <= D; ++i) {
        int o = offset_(i, j);
        float w = barycentric_(i, j);
        const_cast< Eigen::MatrixBase<OutDerived>& >(out).col(j) += w * alpha * values.col(o);
      }
    }
  }

  int numOfFeatures() const {return N_;}
  int numOfLatticePoints() const {return M_;}
  int featureDimenssion() const {return D;}

};

}  // end namespace vp

#endif /* VIDEOPARSING_DENSEGM_PERMUTOHEDRALLATTICE_H_ */
