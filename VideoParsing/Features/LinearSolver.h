/**
 * @file LinearSolver.h
 * @brief LinearSolver
 *
 * @author Abhijit Kundu
 */

#ifndef VIDEOPARSING_FEATURES_LINEARSOLVER_H_
#define VIDEOPARSING_FEATURES_LINEARSOLVER_H_

#include "VideoParsing/Utils/ScopedTimer.h"
#include <iostream>
#include <Eigen/Sparse>

#include <amgcl/amgcl.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <boost/range/algorithm.hpp>


namespace vp {

template<class DerivedA, class DerivedB, class DerivedX>
double computeResidual(const Eigen::SparseMatrixBase<DerivedA>& A,
                       const Eigen::MatrixBase<DerivedB>& B,
                       const Eigen::MatrixBase<DerivedX>& X) {

  typedef typename DerivedB::PlainObject PlainMatrixType;
  PlainMatrixType E = A * X - B;
  return E.squaredNorm();
}

template<class DerivedA, class DerivedB>
typename DerivedB::PlainObject solveViaCholesky(const Eigen::SparseMatrixBase<DerivedA>& A,
                                                const Eigen::MatrixBase<DerivedB>& B) {
  ScopedTimer tmr;
  std::cout << "Solving via SimplicialCholesky\n";

  typedef typename Eigen::SimplicialCholesky<DerivedA> SolverType;
  SolverType solver;

  {
    ScopedTimer tmr("Solver.compute(A) ... ");
    solver.compute(A);
  }

  typename DerivedB::PlainObject X;
  {
    ScopedTimer tmr("Solver.solve(B) ... ");
    X = solver.solve(B);
  }
  std::cout << "Final Energy = " << std::scientific << computeResidual(A, B, X) << "\n";

  std::cout << "Total Time = ";

  return X;
}

template<class DerivedA, class DerivedB>
typename DerivedB::PlainObject solveViaCG(const Eigen::SparseMatrixBase<DerivedA>& A,
                                          const Eigen::MatrixBase<DerivedB>& B,
                                          const int max_iterations = 500) {
  ScopedTimer tmr;

  std::cout << "Solving via ConjugateGradient\n";

  typedef typename Eigen::ConjugateGradient<DerivedA> SolverType;
  SolverType solver;
  solver.setMaxIterations (max_iterations);

  {
    ScopedTimer tmr("Solver.compute(A) ... ");
    solver.compute(A);
  }

  typename DerivedB::PlainObject X;
  {
    ScopedTimer tmr("Running " + std::to_string(max_iterations) + " iterations ... ");
    X = solver.solve(B);
  }

  std::cout << "Final Energy = " << std::scientific << computeResidual(A, B, X) << "\n";

  std::cout << "Total Time = ";

  return X;
}

template<class DerivedA, class DerivedB, class DerivedX>
typename DerivedB::PlainObject solveViaCG(const Eigen::SparseMatrixBase<DerivedA>& A,
                                          const Eigen::MatrixBase<DerivedB>& B,
                                          const Eigen::MatrixBase<DerivedX>& intial_guess,
                                          const int max_iterations = 500) {
  ScopedTimer tmr;

  std::cout << "Solving via ConjugateGradient\n";

  typedef typename Eigen::ConjugateGradient<DerivedA> SolverType;
  SolverType solver;
  solver.setMaxIterations (max_iterations);

  std::cout << "Initial Energy = " << std::scientific << computeResidual(A, B, intial_guess) << "\n";

  {
    ScopedTimer tmr("Solver.compute(A) ... ");
    solver.compute(A);
  }

  typename DerivedB::PlainObject X;
  {
    ScopedTimer tmr("Running " + std::to_string(max_iterations) + " iterations ... ");
    X = solver.solve(B);
  }

  std::cout << "Final Energy = " << std::scientific << computeResidual(A, B, X) << "\n";

  std::cout << "Total Time = ";

  return X;
}


template<class DerivedA, class DerivedB>
typename DerivedB::PlainObject solveViaAMGCL(const Eigen::SparseMatrixBase<DerivedA>& A,
                                             const Eigen::MatrixBase<DerivedB>& B) {
  ScopedTimer tmr;

  std::cout << "Solving via AMGCL\n";

  typedef typename DerivedA::Scalar Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DenseVector;

  static_assert(DerivedA::IsRowMajor == 1, "Needs a row major matrix");

  if(! A.derived().isCompressed() )
    throw std::runtime_error("Needs a compressed matrix");

  if( A.rows() != A.cols() )
    throw std::runtime_error("Needs a square matrix");

  const Eigen::Index n = A.rows();

  // Define the AMG type:
  typedef amgcl::amg<
          amgcl::backend::builtin<Scalar>,
          amgcl::coarsening::ruge_stuben,
          amgcl::relaxation::damped_jacobi
          > AMG;

  // Construct the AMG hierarchy.
  AMG amg(
      boost::make_tuple(
          n, boost::make_iterator_range(A.derived().outerIndexPtr(), A.derived().outerIndexPtr() + n + 1),
          boost::make_iterator_range(A.derived().innerIndexPtr(), A.derived().innerIndexPtr() + A.derived().derived().outerIndexPtr()[n]),
          boost::make_iterator_range(A.derived().valuePtr(), A.derived().valuePtr() + A.derived().outerIndexPtr()[n])));


  // Output some information about the constructed hierarchy:
  std::cout << amg << std::endl;


  // Use BiCGStab as an iterative solver:
  typedef amgcl::solver::bicgstab<amgcl::backend::builtin<double> > Solver;

  // Construct the iterative solver. It needs size of the system to
  // preallocate the required temporary structures:
  Solver solve(n);


  typename DerivedB::PlainObject X(A.rows(), B.cols());
  {
    ScopedTimer tmr("Running bicgstab iterations ... \n");
    for(Eigen::Index c = 0; c < B.cols(); ++c) {
      std::cout << "Solving channel " << c  << " ... " << std::flush;
      std::vector<Scalar> x(n, 0);
      std::vector<Scalar> b(n);
      for(Eigen::Index i=0; i<n; ++i)
        b[i] = B(i, c);

      int iters;
      double resid;
      boost::tie(iters, resid) = solve(amg, b, x);

      std::cout << " Iterations: " << iters
                << " Error: " << resid << std::endl;

      X.col(c) = Eigen::Map<DenseVector>(x.data(), n);
    }

  }

  std::cout << "Final Energy = " << std::scientific << computeResidual(A, B, X) << "\n";

  std::cout << "Total Time = ";

  return X;
}

template<class DerivedA, class DerivedB, class DerivedX>
typename DerivedB::PlainObject solveViaAMGCL(const Eigen::SparseMatrixBase<DerivedA>& A,
                                             const Eigen::MatrixBase<DerivedB>& B,
                                             const Eigen::MatrixBase<DerivedX>& intial_guess) {
  ScopedTimer tmr;

  std::cout << "Solving via AMGCL with initial guess\n";

  typedef typename DerivedA::Scalar Scalar;
  typedef typename DerivedA::StorageIndex StorageIndex;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> DenseVector;

  static_assert(DerivedA::IsRowMajor == 1, "Needs a row major matrix");

  if(! A.derived().isCompressed() )
    throw std::runtime_error("Needs a compressed matrix");

  if( A.rows() != A.cols() )
    throw std::runtime_error("Needs a square matrix");

  std::cout << "Initial Energy = " << std::scientific << computeResidual(A, B, intial_guess) << "\n";

  const Eigen::Index n = A.rows();

  // Define the AMG type:
  typedef amgcl::amg<
          amgcl::backend::builtin<Scalar>,
          amgcl::coarsening::ruge_stuben,
          amgcl::relaxation::damped_jacobi
          > AMG;

  const Scalar *val  = A.derived().valuePtr(); // Values.
  const StorageIndex* col_ind = A.derived().innerIndexPtr(); // Column numbers.
  const StorageIndex* row_ptr = A.derived().outerIndexPtr(); // Row pointers into the above arrays.

  // Construct the AMG hierarchy.
  AMG amg(
      boost::make_tuple(
          n, boost::make_iterator_range(row_ptr, row_ptr + n + 1),
          boost::make_iterator_range(col_ind, col_ind + row_ptr[n]),
          boost::make_iterator_range(val, val + row_ptr[n])));


  // Output some information about the constructed hierarchy:
  std::cout << amg << std::endl;


  // Use BiCGStab as an iterative solver:
  typedef amgcl::solver::bicgstab<amgcl::backend::builtin<double> > Solver;

  // Construct the iterative solver. It needs size of the system to
  // preallocate the required temporary structures:
  Solver solve(n);


  typename DerivedB::PlainObject X(A.rows(), B.cols());
  {
    ScopedTimer tmr("Running bicgstab iterations\n");
    for(Eigen::Index c = 0; c < B.cols(); ++c) {
      std::cout << "Solving channel " << c  << " ... " << std::flush;
      std::vector<Scalar> x(n);
      for(Eigen::Index i=0; i<n; ++i)
        x[i] = intial_guess(i, c);

//      std::vector<Scalar> b(n);
//      for(std::size_t i=0; i<n; ++i)
//        b[i] = B(i, c);

      const Scalar * bptr = &B(0,c);
      int iters;
      double resid;
      boost::tie(iters, resid) = solve(amg, boost::make_iterator_range(bptr, bptr + n), x);

      std::cout << " Iterations: " << iters
                << " Error: " << resid << std::endl;

      X.col(c) = Eigen::Map<DenseVector>(x.data(), n);
    }

  }

  std::cout << "Final Energy = " << std::scientific << computeResidual(A, B, X) << "\n";

  std::cout << "Total Time = ";

  return X;
}


}  // namespace vp

#endif /* VIDEOPARSING_FEATURES_LINEARSOLVER_H_ */
