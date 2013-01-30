/*
Copyright © 2010, Ismion LL
All rights reserved
http://www.ismion.com

Redistribution and use in source and binary forms, with or withou
modification IS NOT permitted without specific prior writte
permission. Further, neither the name of the company, Analytics130
Inc, nor the names of its employees may be used to endorse or promot
products derived from this software without specific prior writte
permission

THIS SOFTWARE IS PROVIDED BY THE Ismion Inc "AS IS" AND AN
EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, TH
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULA
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COMPANY BE LIABLE FO
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIA
DAMAGES INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOOD
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISIN
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF TH
POSSIBILITY OF SUCH DAMAGE
*/
/**
 * @file linear_algebra.h
 *
 * Integration with BLAS and LAPACK and Trilinos.
 */

#ifndef LA_LINEAR_ALGEBRA_H
#define LA_LINEAR_ALGEBRA_H
#include "linear_algebra_defs.h"
#include "fastlib/math/fl_math.h"

namespace fl {
namespace la {

/**
 * @brief Scales the rows of a column-major matrix by a different value for
 * each row.
 * @code
 *   template<Precision>
 *   void ScaleRows(index_t n_rows, index_t n_cols,
 *       const Precision *scales, const Precision *matrix);
 *   // example:
 *   double a[3][3]={{0, 1, 2}, {-1, 4, 2}, {-3, 0, -2}};
 *   double scales[3]={1, -3, 1};
 *   fl::la::Scale(3, 4, scales, a);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param n_rows, number of rows of the matrix
 * @param n_cols, number of columns of the matrix
 * @param scales, an array with the values that scale the rows
 * @param matrix, a column major matrix unfolded in a memory slab
 */

template<typename MatrixType1, typename MatrixType2>
void  inline ScaleRows(const MatrixType1 &d, MatrixType2* X) {
  MatrixType2::ScaleRows(d, X);
}

/**
 * @brief Finds the square root of the dot product of a vector or matrix with itself
 *       (\f$\sqrt{x} \cdot x \f$).
 * @code
 *  template<typename Precision, bool IsVector>
 *  inline Precision LengthEuclidean(const GenMatrix<Precision, IsVector> &x);
 *  // example
 *  fl::la::GenMatrix<float> a;
 *  fl::la::Random(3,1, &a);
 *  float norm = fl::la::LengthEuclidean(a);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param IsVector, boolean variable, for backward compatibility
 * @param x, a matrix
 *
 */

template<typename MatrixType>
inline double LengthEuclidean(const MatrixType &x) {
  return MatrixType::LengthEuclidean(x);
}

/**
 * @brief Finds the dot product of two arrays
 *        (\f$x \cdot y\f$).
 * @code
 *   template<typename Precision, bool IsVector>
 *   Precision Dot(const GenMatrix<Precision, IsVector> &x,
 *                 const GenMatrix<Precision, IsVector> &y);
 *   // example
 *   fl::la::GenMatrix<double> x;
 *   fl::la::GenMatrix<double> y;
 *   fl::la::Random(1, 4, &x);
 *   fl::la::Random(1, 4, &y);
 *   double dot_prod = fl::la::Dot(x, y);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double
 * @param IsVector, template parameter for backward compatibility
 * @param x, a GenMatrix
 * @param y, a GenMatrix, with the same dimensions as x
 */

template<typename MatrixType1, typename MatrixType2>
inline double Dot(const MatrixType1 &x,
    const MatrixType2 &y) {
  return MatrixType1::Dot(x, y);
}

/**
 * @brief Finds the dot product of two arrays
 *        (\f$x \cdot y\f$).
 * @code
 *   template<typename Precision, bool IsVector>
 *   Precision Dot(const GenMatrix<Precision, IsVector> &x,
 *                 const GenMatrix<Precision, IsVector> &W,
 *                 const GenMatrix<Precision, IsVector> &y);
 *   // example
 *   fl::la::GenMatrix<double> x;
 *   fl::la::GenMatris<double> W;
 *   fl::la::GenMatrix<double> y;
 *   fl::la::Random(1, 4, &x);
 *   fl::la::Random(4, 4, &W);
 *   fl::la::Random(1, 4, &y);
 *   double dot_prod = fl::la::Dot(x, W, y);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double
 * @param IsVector, template parameter for backward compatibility
 * @param x, a GenMatrix
 * @param y, a GenMatrix, with the same dimensions as x
 */

template<typename MatrixType1, 
         typename MatrixType2, 
         typename MatrixType3>
inline double Dot(const MatrixType1 &x,
    const MatrixType2 &W,
    const MatrixType3 &y) {
  return MatrixType1::Dot(x, W, y);
}

/**
 * @brief Scales an array in-place by some factor
 * (\f$ x \gets \alpha x\f$).
 * @code
 *  template<typename Precision, bool IsVector>
 *  void ScaleExpert(const Precision alpha, GenMatrix<Precision, IsVector> *x);
 *  //example
 *  fl::la::GenMatrix<double> x;
 *  fl::la::Random(3, 5, &x);
 *  double alpha=2.4;
 *  fl::la::ScaleExpert(alpha, &x);
 * @endcode
 *
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float,double
 * @param IsVector, template parameter for backward compatibility
*/

template<typename MatrixType>
inline void SelfScale(double alpha, MatrixType *x) {
  MatrixType::SelfScale(alpha, x);
}



/**
   * @brief Sets an array to another scaled by some factor
   *        (\f$ y \gets \alpha x\f$).
   * @code
   *  template<typename Precision, MemoryAlloc M, bool IsVector>
   *  void Scale(const Precision alpha, const GenMatrix<Precision, IsVector>
   *       &x, GenMatrix<Precision, IsVector> *y);
   *  //example
   *  fl::la::GenMatrix<double> x;
   *  fl::la::Random(4, 6, &x);
   *  fl::la::GenMatrix<double> y;
   *  double alpha = 3.44;
   * // The call sounds weird, but we used a C++ trick to achieve that
   * // You can use the following syntax works
   *  Scale<fl::la::Init>(alpha, x, &y);
   * // or initializr y first
   * y.Init(4, 6);
   *  Scale<fl::la::Overwrite> (alpha, x, &y);
   * @endcode
   *
   * @param Precision, template parameter for the precision, currently supports
   *        float, double.
   * @param IsVector, bool parametre for bakward compatibility with Vectors
   * @param M, this one is the only one you have to define, it can be
   *           fl::la::Init if you want the function to allocate space for the
   *           result or, fl::la::Overwrite if the result is already initialized
   * @param alpha, the scaling factor
   * @param x, the matrix to be scaled
   * @param y, the scaled matrix
   */

template<MemoryAlloc M>
class Scale {
  public:
    template<typename MatrixType1, typename MatrixType2>
    Scale(double alpha,
          const MatrixType1 &x,
          MatrixType2 *y) {
      typename MatrixType2::template Scale<M>(alpha, x, y);
    }
};


/**
 * @brief Adds a scaled vector to an existing vector
 *        (\f$ y \gets y + \alpha x\f$).
 * @code
 *   template<typename Precision, bool IsVector>
 *   void AddExpert(Precision alpha,
 *                  const GenMatrix<Precision, IsVector> &x,
 *                  GenMatrix<Precision, IsVector> *y);
 *   // example
 *   float alpha
 *   fl::la::GenMatrix<float> x;
 *   fl::la::Random(5, 4, &x);
 *   fl::la::GenMatrix<float> y;
 *   fl::la::Random(5, 4, &y);
 *   fl::la::AddExpert(alpha, x, &y);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param IsVector, boolean template parameter for backward compatibility with
 *                  Vector
 */

template<typename MatrixType1, typename MatrixType2>
inline void AddExpert(const typename MatrixType2::CalcPrecision_t alpha,
                      const MatrixType1 &x,
                      MatrixType2 * const y) {
  MatrixType2::AddExpert(alpha, x, y);
}

template<typename MatrixType1, typename MatrixType2>
inline void AddTo(const typename MatrixType2::CalcPrecision_t alpha,
                  const MatrixType1 &x,
                  MatrixType2 * const y) {
  AddExpert(alpha, x, y);
}


/**
 * @brief Adds a vector to an existing vector
 *        (\f$ y \gets y + x\f$);
 * @code
 *   template<typename Precision, bool IsVector>
 *   void AddTo(const GenMatrix<Precision, IsVector> &x,
 *              GenMatrix<Precision, IsVector> *y);
 *   // example
 *   fl::la::GenMatrix<double> x;
 *   fl::la::Random(7, 8, &x);
 *   fl::la::GenMatrix<double> y;
 *   fl::la::Random(7, 9, &y);
 *   fl::la::AddTo(x, y);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float double
 * @param IsVector, template parameter, for backward compatibility with Vector
 */

template<typename MatrixType1, typename MatrixType2>
inline void AddTo(const MatrixType1 &x,
                  MatrixType2 * const y) {
  MatrixType2::AddTo(x, y);
}

/**
 * @brief Sets a vector to the sum of two vectors
 *        (\f$ z \gets y + x\f$).
 * @code
 * // Add is a function. We use the following trick though
 * // To use it as a constructor of a class.
 * // The Precision and IsVector are deduced automatically
 * // from the syntax. The M (Init, Overwrite) have to be defined
 * // explicitly. The example will shed more light on this issue
 *  template<MemoryAlloc M>
 *  class Add {
 *   public:
 *    template<typename Precision, , bool IsVector>
 *    Add(const GenMatrix<Precision, IsVector> &x,
 *       const GenMatrix<Precision, IsVector> &y,
 *             GenMatrix<Precision, IsVector> *z);
 *  };
 *  // example
 *  fl::la::GenMatrix<double> x;
 *  fl::la::GenMatrix<double> y;
 *  fl::la::Random(5, 6, &x);
 *  fl::la::Random(5, 6, &y);
 *  fl::la::GenMatrix<double> z;
 *  // Add will initialize z
 *  fl::la::Add<fl::la::Init>(x, y, &z);
 *  // alternatively
 *  fl::la::GenMatrix<double> w;
 *  w.Init(5, 6);
 * // Add desn't allocate space for w
 *  fl::la::Add<fl::la::Overwrite>(x, y, &w)/
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. This parameter is automatically deduced from the
 *        function arguments
 * @param IsVector, for backward compatibility with Vector. This parameter is automatically deduced from the
 *        function arguments
 */

template<MemoryAlloc M>
class Add {
  public:
    template<typename MatrixType>
    Add(const MatrixType &x,
        const MatrixType &y,
        MatrixType * const z) {
      MatrixType::Add<M>(x, y, z);
    }
};

/* --- Matrix/Vector Subtraction --- */

/**
 * @brief Subtracts a vector from an existing vector
 *        (\f$ y \gets y - x \f$).
 * @code
 *  template<typename Precision, bool IsVector>
 *  void SubFrom(const GenMatrix<Precision, IsVector> &x,
 *                     GenMatrix<Precision, IsVector> *y);
 *   //example
 *   fl::la::GenMatrix<float> x;
 *   fl::la::GenMatrix<float> y;
 *   fl::la::Random(3, 4, &x);
 *   fl::la::Random(3, 4, &y);
 *   fl::la::SubFrom(x, &y);
 * @endcode
 *
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. The type is automatically deduced by the function
 *        arguments
 * @param IsVector, boolean templated parameter for backward compatibility
 *                  with Vector. It is automatically deduced from the
 *                  function arguments
 */

template<typename MatrixType1, typename MatrixType2>
inline void SubFrom(const MatrixType1 &x,
                    MatrixType2 * const y) {
  MatrixType1::SubFrom(x, y);
}

/**
 * @brief Sets a vector to the difference of two vectors
 *        (\f$ z \gets y - x\f$).
 * @code
 *   // Although Sub is a function it is declared as a class.
 *   // This is because the M argument has to be explicitly
 *   // defined while Precision, and IsVector can be
 *   // deduced automatically by the compiler
 *   // The following example will shed more light
 *   template<MemoryAlloc M>
 *   class Sub {
 *    public:
 *    template<typename Precision, bool IsVector>
 *    Sub(const GenMatrix<Precision, IsVector> &x,
 *        const GenMatrix<Precision, IsVector> &y,
 *              GenMatrix<Precision, IsVector> *z);
 *  };
 *  // example
 *  fl::la::GenMatrix<double> x;
 *  fl::la::GenMatrix<double> y;
 *  fl::la::Random(4, 4, &x);
 *  fl::la::Random(4, 4, &z);
 *  fl::la::GenMatrix z;
 *  // Sub will allocate space for z
 *  fl::la::Sub<fl::la::Init>(x, y, &z);
 *  fl::la::GenMatrix<double> w;
 *  w.Init(4, 4);
 *  // Sub will not allocate space for z
 *  fl::la::Sub<fl::la::Overwrite>(x, y, &z);
 * @endcode
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced from function arguments
 * @param IsVector, bool parameter, for backward compatibility with Vector.
 *                  It is automatically deduced from the function arguments
 * @param M, it can be fl::la::Init if we want the function to initialize the
 *           result, or fl::la::Overwrite if the result has already been initialized
 */
template<MemoryAlloc M>
class Sub {
  public:
    template<typename MatrixType1, 
      typename MatrixType2,
      typename MatrixType3>
    Sub(const MatrixType1 &x,
        const MatrixType2 &y,
        MatrixType3 * const z) {
      typename MatrixType1::template Sub<M>(x, y, z);
    }
};

/* --- Matrix Transpose --- */

/*
 * TODO: These could be an order of magnitude faster if we use the
 * cache-efficient Morton layout matrix transpose algoritihm.
 */

/**
 * @brief Computes a square matrix transpose in-place
 *        (\f$X \gets X'\f$).
 * @code
 *   template<typename Precision>
 *   void TransposeSquare(GenMatrix<Precision, false> *X);
 *   // example
 *   fl::la::GenMatrix<float> x;
 *   fl::la::Random(4, 4, &x);
 *   fl::la::TransposeSquare(&x);
 * @endcode
 *
 * @param Precision, template parameter for the precision, Automatically
 *                   deduced by the function arguments
 */

template<typename MatrixType>
inline void TransposeSquare(MatrixType * const x) {
  MatrixType::Transpose(x);
}

/**
  * @brief Sets a matrix to the transpose of another
  *        (\f$Y \gets X'\f$).
  * @code
  *   // Although Transpose is a function we declare it as a class and
  *   // we use it as a constructor. This is because the template M parameter
  *   // has to be explicitly defined, while the others  can be deduced.
  *   // The following example will shed some light
  *   template<MemoryAlloc M>
  *   class Transpose {
  *    public:
  *     template<typename Precision>
  *     Transpose(const GenMatrix<Precision, false> &X,
  *                     GenMatrix<Precision, false> *Y);
  *   };
  *   // example
  *   fl::la::GenMatrix<double> x;
  *   fl::la::GenMatrix<double> y;
  *   fl::la::Random(4, 6, &x);
  *   fl::la::Transpse<fl::la::init>(x, &y);
  *   // or if we want to initialize on our own
  *   fl::la::GenMatrix w;
  *   w.Init(6, 4);
  *   fl::la::Transpose<fl::la::Overwrite>(x, &y);
  *
  * @endcode
  * @param Precision, template parameter for the precision, automatically
  *                   deduced from the functtion arguments
  * @param M, it can be fl::la::Init if we want the function to initialize the
  *           result, or fl::la::Overwrite if the result has already been initialized
  */
template<MemoryAlloc M>
class Transpose {
  public:
    template<typename MatrixType>
    Transpose(const MatrixType &x, MatrixType * const y) {
      MatrixType::Transpose(x, y);
    }
};


/**
 * @brief Matrix Multiplication, close to the Blas format
 *        (\f$  C = A *B + b * C \f$)
 *        A, B  can be in normal or transpose mode
 * @code
 *  // Although MulExpert is a function, we declare it as a
 *  // class constructor, only because IsTransA, IsTransB
 *  // have to be declared explicitly. The other template
 *  // parameters can be automatically deduced by the function arguments
 *  // The following example will shed some light
 *  template<TransMode IsTransA, TransMode IsTransB>
 *  class MulExpert {
 *   public:
 *    template<typename Precision>
 *    MulExpert(Precision alpha,
 *              const GenMatrix<Precision, false> &a,
 *              const GenMatrix<Precision, false> &b,
 *              Precision beta,
 *              GenMatrix<Precision, false> *c);
 *     template<typename Precision>
 *  MulExpert(const Precision alpha,
 *            const GenMatrix<Precision, false> &A,
 *            const GenMatrix<Precision, true> &x,
 *                  Precision beta,
 *                  GenMatrix<Precision, true> *y);
 *  };
 *  // example
 *  fl::la::GenMatrix<double> a;
 *  fl::la::GenMatrix<double> b;
 *  fl::la::GenMatrix<double> c;
 *  fl::la::Random(4, 6, &a);
 *  fl::la::Random(5, 6, &b);
 *  fl::la::Random(4, 5, &c);
 *  double beta=5.33;
 *  fl::la::MulExpert<fl::la::NoTrans, fl::la::Trans>(a, b, beta, c);
 * @endcode
 *
 * @param IsTransA, it can be fl::la::NoTrans, fl::la::Trans, if we are using
 *                  A or A'
 * @param IsTransB it can be fl::la::NoTrans, fl::la::Trans, if we are using
 *                  B or B'
 * @param Precision, template parameter for the precision, currently supports
 *        float,double. It is automatically deduced from the function arguments
 */

template < TransMode IsTransA, TransMode IsTransB = NoTrans >
class MulExpert {
  public:
    template<typename MatrixType1, typename MatrixType2, typename MatrixType3>
    MulExpert(const typename MatrixType1::CalcPrecision_t alpha,
              const MatrixType1 &a,
              const MatrixType2 &x,
              const typename MatrixType1::CalcPrecision_t beta,
              MatrixType3 * const y) {

      MatrixType1::MulExpert<IsTransA, IsTransB>(a, x, beta, y);
    }
};


/**
 * @brief Matrix Multiplication
 *       (\f$ C = A^{(T)} * B^{(T)}   \f$)
 *       A and B can be in normal or transpose mode
 * @code
 *   // Although Mul is a function, it is being used as a class constructor
 *   // This is because M, IsTransA have to be expliitly declared, while
 *   // the other parameters are automatically deduced by the function
 *   // arguments, The following example will shed some light
 *   template<MemoryAlloc M,
 *            TransMode IsTransA,
 *            TransMode IsTransB>
 *   class Mul {
 *   public:
 *    template<typename Precision,
 *             bool IsVector>
 *    Mul(const GenMatrix<Precision, false> &a,
 *        const GenMatrix<Precision, IsVector> &b,
 *        GenMatrix<Precision, IsVector> *c);
 *   };
 *   // example
 *   fl::la::GenMatrix<double> a;
 *   fl::la::GenMatrix<double> b;
 *   fl::la::GenMatrix<double> c;
 *   fl::la::Random(4, 3, &a);
 *   fl::la::Random(3, 4, &b);
 *   Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(a, b, &c);
 * @endcode
 *
 * @param M, It can be fl::la::Init, or fl::la::Overwrite, depending on whether
 *           we want the result to be initialized by Mul or not.
 * param IsTransA, It can be fl::la::NoTrans or fl::la::Trans, depending on
 *                 whether we want to use A's transpose or not.
 * @param Precision, template parameter for the precision, currently supports
 *        float, double. It is automatically deduced by the funtion arguments
 * @param IsVector, bool parameter for backward compatibility,
 *                  automatically deduced from the function arguments
 *
 */
template < MemoryAlloc M,
TransMode IsTransA = NoTrans,
TransMode IsTransB = NoTrans >
class Mul {
  public:
    template<typename MatrixType1, typename MatrixType2, typename MatrixType3>
    Mul(const MatrixType1 &a, const MatrixType2 &b, MatrixType2 * const c) {
      MatrixType1::Mul<M, IsTransA, IsTransB>(a, b, c);
    }
};

/**
  * Finds the Euclidean distance squared between two vectors.
  */
template<typename MatrixType1, typename MatrixType2>
inline typename MatrixType1::CalcPrecision_t DistanceSqEuclidean(const MatrixType1& x,
    const MatrixType2& y) {
  return MatrixType1::DistanceSqEuclidean(x, y);
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
template<int t_pow>
class RawLMetric {
  public:
    template<typename PointType1, typename PointType2, typename CalcPrecisionType>
    RawLMetric(const PointType1 &va,
               const PointType2 &vb, CalcPrecisionType *result)  {
      typename PointType1::template RawLMetric<t_pow>(va, vb, result);
    }

    template < typename PointType1, typename PointType2, typename WeightPointType,
    typename CalcPrecisionType >
    RawLMetric(const WeightPointType &w,
               const PointType1 &va,
               const PointType2 &vb, CalcPrecisionType *result)  {
      typename PointType1::template RawLMetric<t_pow>(w, va, vb, result);
    }
};
/**
 * Finds an L_p metric distance AND performs the root
 * at the end.
 *
 * @param t_pow the power each distance calculatin is raised to
 * @param length the length of the vectors
 * @param va first vector
 * @param vb second vector
 */
template<int t_pow>
class LMetric {
  public:
    template<typename MatrixType1, typename MatrixType2>
    LMetric(const MatrixType1 &x, const MatrixType2 &y,
            typename MatrixType1::Precision_t * const result) {
      *result = MatrixType1::LMetric(x.n_elements(), x.ptr(), y.ptr());
    }
};

/** Finds the trace of the matrix.
 *  Trace(A) is the sum of the diagonal elements
 */
template<typename MatrixType>
inline void Trace(const MatrixType &a,
                  typename MatrixType::Precision_t *const trace) {
  *trace = Trace(a);
}

/**
 * @brief Entrywise element multiplication of two matrices
 * Also known as the Hadamard product
 * In Matlab this is the .* operator
 * (\f$ C \gets A .* B\f$).
 * @code
 *   // example
 *   fl::la::GenMatrix<double> a;
 *   fl::la::GenMatrix<double> b;
 *   fl::la::Random(3, 1, &a);
 *   fl::la::Random(3, 1, &b);
 *   fl::la::GenMatrix<double> c;
 *   // if c is not initialized, DotMul will allocate the space for c
 *   fl::la::DotMul<fl::la::Init(a, b, &c);
 *   // now that it is initialized
 *   fl::la::DotMul>fl::la::Overwrite>(a, b, &c);
 *
 * @endcode
 * @param A, matrix
 * @param B, matrix
 * @param C, matrix C=A.*B
 */

template<MemoryAlloc MemAlloc>
class DotMul {
  public:
    template<typename MatrixType1, typename MatrixType2, typename MatrixType3>
    DotMul(const MatrixType1 &a,
           const MatrixType2 &b,
           MatrixType3 * const c) {
      typename MatrixType1::template DotMul<MemAlloc>(a, b, c);
    }
};

/**
 * @brief Entrywise element multiplication of two matrices
 * Also known as the Hadamard product
 * In Matlab this is the .* operator
 * (\f$ A \gets A .* B\f$).
  * @code
 *   // example
 *   fl::la::GenMatrix<float> a;
 *   fl::la::GenMatrix<float> b;
 *   fl::la::Random(3, 4, &a);
 *   fl::la::Random(3, 4, &b;
 *   fl::la::DotMulTo(&a, b);
 * @endcode
 * @param A, matrix
 * @param B, matrix
 */

template<typename MatrixType1, typename MatrixType2>
void DotMulTo(const MatrixType1 &b,
              MatrixType2 * const a) {
  MatrixType1::DotMulTo(b, a);
}


/**
 * @brief it sums all the elements of matrix A
 * (\f$ sum(sum(A))\f$)
 * @code
 *   // example
 *   fl::la::GenMatrix<float> a;
 *   fl::la::Random(100, 1000, &a);
 *   double mysum;
 *   Sum(a, &mysum);
 * @endcode
 */
class Sum {
  public:
    template<typename MatrixType>
    Sum(const MatrixType &a, typename MatrixType::CalcPrecision_t* const sum) {
      typename MatrixType::Sum(a, sum);
    }
};

/**
 * @brief it sums all the columns of matrix A
 *        (\f$ sum(A)\f$)
 * @code
 *   fl::la::GenMatrix<float> a;
 *   fl::la::Random(40, 70, &a);
 *   fl::la::GenMatrix< col_sums;
 *   fl::la::SumCols<fl::la::Init>(a, &col_sums);
 * @endcode
 */
template<MemoryAlloc MemAlloc>
class SumCols {
  public:
    template<typename MatrixType1, typename MatrixType2>
    SumCols(const MatrixType1 &a,
            MatrixType2* const col_sums) {
      MatrixType1::SumCols<MemAlloc>(a, col_sums);
    }
};

/**
 * @brief it sums all the rows of matrix A
 *        (\f$ sum(A,2)\f$)
 * @code
 *   fl::la::GenMatrix<double> a;
 *   fl::la::Random(50, 60, &a);
 *   fl::la::GenMatrix<long double> row_sums;
 *   fl::la::SumRows<fl::la::Init>(a, &row_sums);
 * @endcode
 */
template<MemoryAlloc MemAlloc>
class SumRows {
  public:
    template<typename MatrixType1, typename MatrixType2>
    SumRows(const MatrixType1 &a,
            MatrixType2* const row_sums) {
      MatrixType1::SuumRows(a, row_sums);
    }
};


} // namespace la
} // namespace fl

#endif
