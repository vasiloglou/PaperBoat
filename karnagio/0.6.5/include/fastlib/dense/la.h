/*
Copyright © 2010, Ismion Inc
All rights reserved.
http://www.ismion.com/

Redistribution and use in source and binary forms, with or without
modification IS NOT permitted without specific prior written
permission. Further, neither the name of the company, Ismion
Inc, nor the names of its employees may be used to endorse or promote
products derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE Ismion Inc "AS IS" AND ANY
EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COMPANY BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
// Copyright 2007 Georgia Institute of Technology. All rights reserved.
// ABSOLUTELY NOT FOR DISTRIBUTION
/**
 * @file la.h
 *
 * Core routines for linear algebra.
 *
 * See uselapack.h for more linear algebra routines.
 */

#ifndef LINEAR_H
#define LINEAR_H

#include "matrix.h"
#include "uselapack.h"
#include "utilities.h"

#include "fastlib/math/fl_math.h"

#include <math.h>

namespace fl {
namespace la {
/**
 * Finds the Euclidean distance squared between two vectors.
 */
template<typename Precision>
inline Precision DistanceSqEuclidean(
  index_t length, const Precision *va, const Precision *vb) {
  Precision s = 0;
  do {
    Precision d = (*va) - (*vb);
    va++;
    vb++;
    s += d * d;
  }
  while (--length);
  return s;
}
/**
 * Finds the Euclidean distance squared between two vectors.
 */
template<typename Precision, bool IsVector>
inline Precision DistanceSqEuclidean(const GenMatrix<Precision, IsVector>& x,
                                     const GenMatrix<Precision, IsVector>& y) {
  DEBUG_SAME_SIZE(x.length(), y.length());
  return DistanceSqEuclidean(x.length(), x.ptr(), y.ptr());
}
/**
 * Finds an L_p metric distance except doesn't perform the root
 * at the end.
 *
 * @param t_pow the power each distance calculatin is raised to
 * @param length the length of the vectors
 * @param va first vector
 * @param vb second vector
 */
template<typename PointType, int t_pow>
inline typename PointType::CalcPrecision_t RawLMetric(const PointType &va,
    const PointType &vb) {
  DEBUG_ASSERT(va.size() == vb.size());
  size_t length = va.size();
  typedef typename PointType::CalcPrecision_t CalcPrecision_t;
  CalcPrecision_t s = 0;
  index_t i = 0;
  do {
    CalcPrecision_t d = va[i] - vb[i];
    i++;
    s += math::PowAbs<CalcPrecision_t, t_pow, 1>(d);
  }
  while (--length);
  return s;
}
/**
 * Finds an L_p metric distance AND performs the root
 * at the end.
 *
 * @param t_pow the power each distance calculatin is raised to
 * @param length the length of the vectors
 * @param va first vector
 * @param vb second vector
 */
template<typename Precision, int t_pow>
inline Precision LMetric(
  index_t length, const Precision *va, const Precision *vb) {
  return math::Pow<Precision, 1, t_pow>(RawLMetric<Precision, t_pow>(length, va, vb));
}
/** Finds the trace of the matrix.
 *  Trace(A) is the sum of the diagonal elements
 */
template<typename Precision, bool IsVector>
inline Precision Trace(GenMatrix<Precision, IsVector> &a) {
  // trace has meaning only for square matrices
  DEBUG_SAME_SIZE(a.n_cols(), a.n_rows());
  Precision trace = 0;
  for (index_t i = 0; i < a.n_cols(); i++) {
    trace += a.get(i, i);
  }
  return trace;
}


/** Solves the classic least square problem y=x*a
 *  where y  is N x r
 *        x  is N x m
 *        a  is m x r
 *  We require that N >= m
 *  a should not be initialized
 */
template<MemoryAlloc M>
class LeastSquareFit {
  public:
    template<typename Precision, bool IsVector>
    LeastSquareFit(GenMatrix<Precision, IsVector> &y,
                   GenMatrix<Precision, false> &x,
                   GenMatrix<Precision, IsVector> *a) {
      DEBUG_SAME_SIZE(y.n_rows(), x.n_rows());
      DEBUG_ASSERT(x.n_rows() >= x.n_cols());
      GenMatrix<Precision, IsVector> r_xy_mat;
      GenMatrix<Precision, IsVector> r_xx_mat;
      fl::la::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(x, x, &r_xx_mat);
      fl::la::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(x, y, &r_xy_mat);
      success_t success = fl::la::Solve<M>(r_xx_mat, r_xy_mat, a).success;
      if unlikely(success != SUCCESS_PASS) {
        if (success == SUCCESS_FAIL) {
          FATAL("Least square fit failed \n");
        }
        else {
          NONFATAL("Least square fit returned a warning \n");
        }
      }
      return ;
    }
    success_t success;
};
/** Solves the classic least square problem y=x'*a
 *  where y  is N x r
 *        x  is m x N
 *        a  is m x r
 *  We require that N >= m
 *  a should not be initialized
 */
template<MemoryAlloc M>
class LeastSquareFitTrans {
  public:
    template<typename Precision, bool IsVector>
    LeastSquareFitTrans(GenMatrix<Precision, IsVector> &y,
                        GenMatrix<Precision, false> &x,
                        GenMatrix<Precision, IsVector> *a) {
      DEBUG_SAME_SIZE(y.n_rows(), x.n_cols());
      DEBUG_ASSERT(x.n_cols() >= x.n_rows());
      GenMatrix<Precision, false> r_xy_mat;
      GenMatrix<Precision, false> r_xx_mat;
      fl::la::Mul<fl::la::Init, fl::la::NoTrans, fl::la::Trans>(x, x, &r_xx_mat);
      fl::la::Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(x, y, &r_xy_mat);
      success = fl::la::Solve<M>(r_xx_mat, r_xy_mat, a);
      if unlikely(success != SUCCESS_PASS) {
        if (success == SUCCESS_FAIL) {
          FATAL("Least square fit failed \n");
        }
        else {
          NONFATAL("Least square fit returned a warning \n");
        }
      }
      return ;
    }
    success_t success;
};
}; //namespace la
}; //namespace fl
#endif
