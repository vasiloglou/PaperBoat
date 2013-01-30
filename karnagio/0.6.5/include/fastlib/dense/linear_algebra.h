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
#ifndef FL_LITE_FASTLIB_DENSE_LINEAR_ALGEBRA_H_
#define FL_LITE_FASTLIB_DENSE_LINEAR_ALGEBRA_H_

/**
 * @file linear_algebra.h
 *
 */

#include "fastlib/base/base.h"
#include "cppblas.h"
#include "cpplapack.h"
#include "linear_algebra_aux.h"
#include "boost/scoped_array.hpp"
#include "fastlib/traits/fl_traits.h"

namespace fl {
namespace  dense {
template<typename Precision, bool IsVector>
class Matrix;

}
}

#define DEBUG_VECSIZE(a, b) \
  DEBUG_SAME_SIZE((a).length(), (b).length())
#define DEBUG_MATSIZE(a, b) \
  DEBUG_SAME_SIZE((a).n_rows(), (b).n_rows());\
  DEBUG_SAME_SIZE((a).n_cols(), (b).n_cols())
#define DEBUG_MATSQUARE(a) \
  DEBUG_SAME_SIZE((a).n_rows(), (a).n_cols())

/*
 * TODO: this could an overhaul; LAPACK failures may mean either input
 * problems or just some linear algebra result (e.g. singular matrix).
 *
 * Perhaps extend success_t?
 */
#define SUCCESS_FROM_LAPACK(info) \
  (likely((info) == 0) ? SUCCESS_PASS : SUCCESS_FAIL)

/**
 *  @brief Linear-algebra routines.
 *
 * This encompasses most basic real-valued vector and matrix math.
 * Most functions are written in a similar style, so after using a few it
 * should be clear how others are used.  For instance, input arguments
 * come first, and the output arguments are last.
 *
 * Many functions have several versions:
 * The functions are all templatized in terms of precision
 * Another template parameter is the initialization of the result
 * So in some cases we want to reuse the same matrix, so it is not necessary
 * to reallocate them
 *
 * All the algorithms are ignorant of Matrix/Vector
 *
 * @code
 * void SomeExampleCode<float, fl::la::Init>(const Matrix& a, const Matrix& b) {
 *   Matrix c;
 *   // c must not be initialized
 *   fl::la::Mul<PrecisionType, fl::la::Trans, fl::la::NoTrans, fl::la::Init>(a, b, &c);
 * }
 * @endcode
 * The enums fl::la::Trans, fl::la::NoTrans are used to declare if the matrix
 * will be used in the transpose or non-transpose mode
 *
 * The enums fl::la::Init, fl::la::Overwrite state whether the resulting
 * container is going to be initialized by the function or if it is
 * already initialized
 *
 * - The Expert version, or power-user version.  We recommend you avoid this
 * version whenever possible.  These are the closest mapping to the LAPACK
 * or BLAS routine.  These will require that memory for destinations is
 * already allocated but the memory itself is not initalized.  These usually
 * take a Matrix or Vector pointer, or sometimes just a pointer to a slab of
 * doubles/floats.  Sometimes, one of the "inputs" is completely destroyed while
 * LAPACK performs its computations.
 *
 *
 * FINAL WARNING:
 * With all of these routines, be very careful about passing in an argument
 * as both a source and destination -- usually this is an error and
 * will result in garbage results, except for simple
 * addition/scaling/subtraction.
 */


namespace fl {
namespace dense {

class ops {
  public:
    /**
     * @brief Scales the rows of a column-major matrix by a different value for
     * each row.
     * @code
     *   template<PrecisionType>
     *   void ScaleRows(index_t n_rows, index_t n_cols,
     *       const PrecisionType *scales, const PrecisionType *matrix);
     *   // example:
     *   double a[3][3]={{0, 1, 2}, {-1, 4, 2}, {-3, 0, -2}};
     *   double scales[3]={1, -3, 1};
     *   fl::la::Scale(3, 4, scales, a);
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     * @param n_rows, number of rows of the matrix
     * @param n_cols, number of columns of the matrix
     * @param scales, an array with the values that scale the rows
     * @param matrix, a column major matrix unfolded in a memory slab
     */
    template<typename PrecisionType>
    static inline void ScaleRows(index_t n_rows, index_t n_cols,
                                 const PrecisionType *scales, PrecisionType *matrix) {
      do {
        for (index_t i = 0; i < n_rows; i++) {
          matrix[i] *= scales[i];
        }
        matrix += n_rows;
      }
      while (--n_cols);
    }

    /**
     * @brief Finds the square root of the dot product of a vector with itself.
     */
    template<typename PrecisionType>
    static inline PrecisionType LengthEuclidean(index_t length, const PrecisionType *x) {
      return CppBlas<PrecisionType>::nrm2(length, x, 1);
    }

    /**
     * @brief Finds the square root of the dot product of a vector or matrix with itself
     *       (\f$\sqrt{x} \cdot x \f$).
     * @code
     *  template<typename PrecisionType, bool IsVectorBool>
     *  inline PrecisionType LengthEuclidean(const Matrix<PrecisionType, IsVectorBool> &x);
     *  // example
     *  fl::la::Matrix<float> a;
     *  fl::la::Random(3,1, &a);
     *  float norm = fl::la::LengthEuclidean(a);
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     * @param IsVectorBool, boolean variable, for backward compatibility
     * @param x, a matrix
     *
     */
    template<typename PrecisionType, bool IsVectorBool>
    static inline PrecisionType LengthEuclidean(const Matrix<PrecisionType, IsVectorBool> &x) {
      return LengthEuclidean<PrecisionType>(x.length(), x.ptr());
    }
    /**
      * @brief Finds the dot-product of two arrays
      * (\f$\vec{x} \cdot \vec{y}\f$).
      */
    template<typename PrecisionType>
    static inline long double Dot(index_t length, const PrecisionType *x, const PrecisionType *y) {
      return CppBlas<PrecisionType>::dot(length, x, 1, y, 1);
    }

    /**
     * @brief Finds the dot product of two arrays
     *        (\f$x \cdot y\f$).
     * @code
     *   template<typename PrecisionType, bool IsVectorBool>
     *   PrecisionType Dot(const Matrix<PrecisionType, IsVectorBool> &x,
     *                 const Matrix<PrecisionType, IsVectorBool> &y);
     *   // example
     *   fl::la::Matrix<double> x;
     *   fl::la::Matrix<double> y;
     *   fl::la::Random(1, 4, &x);
     *   fl::la::Random(1, 4, &y);
     *   double dot_prod = fl::la::Dot(x, y);
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double
     * @param IsVectorBool, template parameter for backward compatibility
     * @param x, a Matrix
     * @param y, a Matrix, with the same dimensions as x
     */
    template<typename PrecisionType, bool IsBoolVector>
    static inline long double Dot(const Matrix<PrecisionType, IsBoolVector> &x,
                                     const Matrix<PrecisionType, IsBoolVector> &y) {
      DEBUG_SAME_SIZE(x.length(), y.length());
      return Dot<PrecisionType>(x.length(), x.ptr(), y.ptr());
    }
    /**
      * @brief Finds the weighted dot-product of two arrays
      * (\f$\vec{x} \mat{W} \vec{y}\f$).
      */
    template<typename PrecisionType, bool IsBoolVector>
    static inline long double Dot(const Matrix<PrecisionType, IsBoolVector> &x,
                                  const Matrix<PrecisionType, false> &W,
                                  const Matrix<PrecisionType, IsBoolVector> &y) {
      DEBUG_SAME_SIZE(x.length(), W.n_rows());
      DEBUG_SAME_SIZE(y.length(), W.n_cols());
      long double result=0;
      for(size_t i=0; i<W.n_cols(); ++i) {
        result+=Dot<PrecisionType>(x.ptr(), W.ptr()+i*W.n_rows())*y[i]; 
      } 
      return result;
    }
    
    /**
      * @brief Finds the weighted dot-product of two arrays
      * (\f$\vec{x} \mat{W} \vec{y}\f$).
      * W is a diagonal matrix
      */
    template<typename PrecisionType, bool IsBoolVector>
    static inline long double Dot(const Matrix<PrecisionType, IsBoolVector> &x,
                                  const Matrix<PrecisionType, true> &W,
                                  const Matrix<PrecisionType, IsBoolVector> &y) {
      DEBUG_SAME_SIZE(x.length(), W.length());
      DEBUG_SAME_SIZE(y.length(), W.length());
      long double result=0;
      for(size_t i=0; i<W.length(); ++i) {
        result+=x[i]*W[i]*y[i]; 
      } 
      return result;
    }
    /**
     *  @brief Updates the outer product of a vector and itself
     *  (\f$ A \gets \alpha x x^T +A)
     */
    template<typename PrecisionType>
    static inline void SelfOuterUpdate(double alpha, 
        index_t length,
        const PrecisionType *x, PrecisionType *A) {
      CppBlas<PrecisionType>::syr('L', length, alpha, x, 1, A, 1);
      for(index_t i=0; i<length; ++i) {
        for(index_t j=0; j<i; ++j) {
          *(A+i+j*length)=*(A+j+i*length);
        }
      }
    }
   
    template<typename PrecisionType1,
             typename PrecisionType2,
             bool IsBoolVector>
    static inline void SelfOuterUpdate(double alpha, 
        const Matrix<PrecisionType1, IsBoolVector> &x, 
        Matrix<PrecisionType2, false> *A) {
      DEBUG_SAME_SIZE(x.length(), A->n_rows());
      DEBUG_SAME_SIZE(x.length(), A->n_cols());
      SelfOuterUpdate(alpha, x.length(), x.ptr(), 
          A->ptr());
    }



    /**
     * @brief Scales an array in-place by some factor
     * (\f$\vec{x} \gets \alpha \vec{x}\f$).
     */
    template<typename PrecisionType>
    static inline void ScaleExpert(index_t length, PrecisionType alpha, PrecisionType *x) {
      CppBlas<PrecisionType>::scal(length, alpha, x, 1);
    }

    /**
     * @brief Scales an array in-place by some factor
     * (\f$ x \gets \alpha x\f$).
     * @code
     *  template<typename PrecisionType, bool IsVectorBool>
     *  void ScaleExpert(const PrecisionType alpha, Matrix<PrecisionType, IsVectorBool> *x);
     *  //example
     *  fl::la::Matrix<double> x;
     *  fl::la::Random(3, 5, &x);
     *  double alpha=2.4;
     *  fl::la::ScaleExpert(alpha, &x);
     * @endcode
     *
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float,double
     * @param IsVectorBool, template parameter for backward compatibility
    */
    template<typename PrecisionType, bool IsVectorBool>
    static inline void ScaleExpert(const PrecisionType alpha, Matrix<PrecisionType, IsVectorBool> *x) {
      ScaleExpert<PrecisionType>(x->length(), alpha, x->ptr());
    }

    template<typename PrecisionType, bool IsVectorBool>
    static inline void SelfScale(const PrecisionType alpha, Matrix<PrecisionType, IsVectorBool> *x) {
      ScaleExpert<PrecisionType>(x->length(), alpha, x->ptr());
    }


    /**
     * @brief Scales each row of the matrix to a different scale.
     *         X <- diag(d) * X
     * @code
     *  template<typename PrecisionType, bool IsVectorBool1, bool IsVectorBool2 >
     *  void ScaleRows(const Matrix<PrecisionType, IsVectorBool1>& d,
     *                       Matrix<PrecisionType, IsVectorBool2> *X);
     *  //example
     *  fl::la::Matrix<float> x;
     *  fl::la::Random(4,5, &x);
     *  fl::la::Matrix<fload> d;
     *  fl::la::Random(4, 1, &d); // since d must be one dimensional n_cols must
     *                            // be one. It must be a column vector
     *  fl::la::ScaleRows(d, &x);
     *  // we could alternative declare d
     *   Matrix<float, true> d;
     *
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     * @param IsVectorBool1, IsVectorBool2, boolean parameters for backward compatibility
     * @param d a length-M vector with each value corresponding
     * @param X the matrix to scale
     */
    template<typename PrecisionType, bool IsVectorBool1, bool IsVectorBool2 >
    static inline void ScaleRows(const Matrix<PrecisionType, IsVectorBool1>& d,
                                 Matrix<PrecisionType, IsVectorBool2> *X) {
      DEBUG_SAME_SIZE(d.n_cols(), 1);
      DEBUG_SAME_SIZE(d.n_rows(), X->n_rows());
      ScaleRows<PrecisionType>(d.n_rows(), X->n_cols(), d.ptr(), X->ptr());
    }
    /**
      * @brief Sets an array to another scaled by some factor
      *        (\f$ y \gets \alpha x\f$).
      * @code
      *  template<typename PrecisionType, MemoryAlloc M, bool IsVectorBool>
      *  void Scale(const PrecisionType alpha, const Matrix<PrecisionType, IsVectorBool>
      *       &x, Matrix<PrecisionType, IsVectorBool> *y);
      *  //example
      *  fl::la::Matrix<double> x;
      *  fl::la::Random(4, 6, &x);
      *  fl::la::Matrix<double> y;
      *  double alpha = 3.44;
      * // The call sounds weird, but we used a C++ trick to achieve that
      * // You can use the following syntax works
      *  Scale<fl::la::Init>(alpha, x, &y);
      * // or initializr y first
      * y.Init(4, 6);
      *  Scale<fl::la::Overwrite> (alpha, x, &y);
      * @endcode
      *
      * @param PrecisionType, template parameter for the precision, currently supports
      *        float, double.
      * @param IsVectorBool, bool parametre for bakward compatibility with Vectors
      * @param M, this one is the only one you have to define, it can be
      *           fl::la::Init if you want the function to allocate space for the
      *           result or, fl::la::Overwrite if the result is already initialized
      * @param alpha, the scaling factor
      * @param x, the matrix to be scaled
      * @param y, the scaled matrix
      */
    template<fl::la::MemoryAlloc M>
    class Scale {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        Scale(const PrecisionType alpha, const Matrix<PrecisionType, IsVectorBool>
              &x, Matrix<PrecisionType, IsVectorBool> *y) {
          AllocationTrait<M>::Init(x.n_rows(), x.n_cols(), y);
          DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
          DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
          y->CopyValues(x);
          ScaleExpert(x.length(), alpha, y->ptr());
        }
    };
    /**
    * @brief Adds a scaled array to an existing array
    * (\f$\vec{y} \gets \vec{y} + \alpha \vec{x}\f$).
    */
    template<typename PrecisionType>
    static inline void AddExpert(index_t length,
                                 PrecisionType alpha, const PrecisionType *x, PrecisionType *y) {
      CppBlas<PrecisionType>::axpy(length, alpha, x, 1, y, 1);
    }

    /**
     * @brief Sets an array to the sum of two arrays
     * (\f$\vec{z} \gets \vec{y} + \vec{x}\f$).
     */
    template<typename PrecisionType>
    static inline void AddExpert(index_t length,
                                 const PrecisionType *x, const PrecisionType *y, PrecisionType *z) {
      ::memcpy(z, y, length * sizeof(PrecisionType));
      AddExpert<PrecisionType>(length, 1.0, x, z);
    }


    /**
     * @brief Adds a scaled vector to an existing vector
     *        (\f$ y \gets y + \alpha x\f$).
     * @code
     *   template<typename PrecisionType, bool IsVectorBool>
     *   void AddExpert(PrecisionType alpha,
     *                  const Matrix<PrecisionType, IsVectorBool> &x,
     *                  Matrix<PrecisionType, IsVectorBool> *y);
     *   // example
     *   float alpha
     *   fl::la::Matrix<float> x;
     *   fl::la::Random(5, 4, &x);
     *   fl::la::Matrix<float> y;
     *   fl::la::Random(5, 4, &y);
     *   fl::la::AddExpert(alpha, x, &y);
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     * @param IsVectorBool, boolean template parameter for backward compatibility with
     *                  Vector
     */
    template<typename PrecisionType, bool IsVectorBool>
    static inline void AddExpert(PrecisionType alpha,
                                 const Matrix<PrecisionType, IsVectorBool> &x,
                                 Matrix<PrecisionType, IsVectorBool> *y) {
      DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
      DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
      AddExpert(x.length(), alpha, x.ptr(), y->ptr());
    }
    /* --- Matrix/Vector Addition --- */

    /**
     * @brief Adds a vector to an existing vector
     *        (\f$ y \gets y + x\f$);
     * @code
     *   template<typename PrecisionType, bool IsVectorBool>
     *   void AddTo(const Matrix<PrecisionType, IsVectorBool> &x,
     *              Matrix<PrecisionType, IsVectorBool> *y);
     *   // example
     *   fl::la::Matrix<double> x;
     *   fl::la::Random(7, 8, &x);
     *   fl::la::Matrix<double> y;
     *   fl::la::Random(7, 9, &y);
     *   fl::la::AddTo(x, y);
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     * @param IsVectorBool, template parameter, for backward compatibility with Vector
     */
    template<typename PrecisionType, bool IsVectorBool>
    static inline void AddTo(const Matrix<PrecisionType, IsVectorBool> &x,
                             Matrix<PrecisionType, IsVectorBool> *y) {
      DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
      DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
      AddExpert(PrecisionType(1.0), x, y);
    }

    /**
     * @brief Sets a vector to the sum of two vectors
     *        (\f$ z \gets y + x\f$).
     * @code
     * // Add is a function. We use the following trick though
     * // To use it as a constructor of a class.
     * // The PrecisionType and IsVectorBool are deduced automatically
     * // from the syntax. The M (Init, Overwrite) have to be defined
     * // explicitly. The example will shed more light on this issue
     *  template<MemoryAlloc M>
     *  class Add {
     *   public:
     *    template<typename PrecisionType, , bool IsVectorBool>
     *    Add(const Matrix<PrecisionType, IsVectorBool> &x,
     *       const Matrix<PrecisionType, IsVectorBool> &y,
     *             Matrix<PrecisionType, IsVectorBool> *z);
     *  };
     *  // example
     *  fl::la::Matrix<double> x;
     *  fl::la::Matrix<double> y;
     *  fl::la::Random(5, 6, &x);
     *  fl::la::Random(5, 6, &y);
     *  fl::la::Matrix<double> z;
     *  // Add will initialize z
     *  fl::la::Add<fl::la::Init>(x, y, &z);
     *  // alternatively
     *  fl::la::Matrix<double> w;
     *  w.Init(5, 6);
     * // Add desn't allocate space for w
     *  fl::la::Add<fl::la::Overwrite>(x, y, &w)/
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. This parameter is automatically deduced from the
     *        function arguments
     * @param IsVectorBool, for backward compatibility with Vector. This parameter is automatically deduced from the
     *        function arguments
     */
    template<fl::la::MemoryAlloc M>
    class Add {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        Add(const Matrix<PrecisionType, IsVectorBool> &x,
            const Matrix<PrecisionType, IsVectorBool> &y,
            Matrix<PrecisionType, IsVectorBool> *z) {
          DEBUG_SAME_SIZE(x.n_rows(), y.n_rows());
          DEBUG_SAME_SIZE(x.n_cols(), y.n_cols());
          AllocationTrait<M>::Init(x.n_rows(), x.n_cols(), z);
          AddExpert(x.length(), x.ptr(), y.ptr(), z->ptr());
        }
    };


    /**
     * @brief Subtracts an array from an existing array
     * (\f$\vec{y} \gets \vec{y} - \vec{x}\f$).
     */
    template<typename PrecisionType>
    static inline void SubFrom(index_t length, const PrecisionType *x, PrecisionType *y) {
      AddExpert<PrecisionType>(length, -1.0, x, y);
    }

    /**
     * @brief Sets an array to the difference of two arrays
     * (\f$\vec{x} \gets \vec{y} - \vec{x}\f$).
     */
    template<typename PrecisionType>
    static inline void SubExpert(index_t length,
                                 const PrecisionType *x, const PrecisionType *y, PrecisionType *z) {
      ::memcpy(z, y, length * sizeof(PrecisionType));
      SubFrom<PrecisionType>(length, x, z);
    }
    /* --- Matrix/Vector Subtraction --- */

    /**
     * @brief Subtracts a vector from an existing vector
     *        (\f$ y \gets y - x \f$).
     * @code
     *  template<typename PrecisionType, bool IsVectorBool>
     *  void SubFrom(const Matrix<PrecisionType, IsVectorBool> &x,
     *                     Matrix<PrecisionType, IsVectorBool> *y);
     *   //example
     *   fl::la::Matrix<float> x;
     *   fl::la::Matrix<float> y;
     *   fl::la::Random(3, 4, &x);
     *   fl::la::Random(3, 4, &y);
     *   fl::la::SubFrom(x, &y);
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. The type is automatically deduced by the function
     *        arguments
     * @param IsVectorBool, boolean templated parameter for backward compatibility
     *                  with Vector. It is automatically deduced from the
     *                  function arguments
     */
    template<typename PrecisionType, bool IsVectorBool>
    static inline void SubFrom(const Matrix<PrecisionType, IsVectorBool> &x,
                               Matrix<PrecisionType, IsVectorBool> *y) {
      DEBUG_SAME_SIZE(x.n_rows(), y->n_rows());
      DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
      SubFrom(x.length(), x.ptr(), y->ptr());
    }

    /**
     * @brief Sets a vector to the difference of two vectors
     *        (\f$ z \gets y - x\f$).
     * @code
     *   // Although Sub is a function it is declared as a class.
     *   // This is because the M argument has to be explicitly
     *   // defined while PrecisionType, and IsVectorBool can be
     *   // deduced automatically by the compiler
     *   // The following example will shed more light
     *   template<MemoryAlloc M>
     *   class Sub {
     *    public:
     *    template<typename PrecisionType, bool IsVectorBool>
     *    Sub(const Matrix<PrecisionType, IsVectorBool> &x,
     *        const Matrix<PrecisionType, IsVectorBool> &y,
     *              Matrix<PrecisionType, IsVectorBool> *z);
     *  };
     *  // example
     *  fl::la::Matrix<double> x;
     *  fl::la::Matrix<double> y;
     *  fl::la::Random(4, 4, &x);
     *  fl::la::Random(4, 4, &z);
     *  fl::la::Matrix z;
     *  // Sub will allocate space for z
     *  fl::la::Sub<fl::la::Init>(x, y, &z);
     *  fl::la::Matrix<double> w;
     *  w.Init(4, 4);
     *  // Sub will not allocate space for z
     *  fl::la::Sub<fl::la::Overwrite>(x, y, &z);
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced from function arguments
     * @param IsVectorBool, bool parameter, for backward compatibility with Vector.
     *                  It is automatically deduced from the function arguments
     * @param M, it can be fl::la::Init if we want the function to initialize the
     *           result, or fl::la::Overwrite if the result has already been initialized
     */
    template<fl::la::MemoryAlloc M>
    class Sub {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        Sub(const Matrix<PrecisionType, IsVectorBool> &x,
            const Matrix<PrecisionType, IsVectorBool> &y,
            Matrix<PrecisionType, IsVectorBool> *z) {
          DEBUG_SAME_SIZE(x.n_rows(), y.n_rows());
          DEBUG_SAME_SIZE(x.n_cols(), y.n_cols());
          AllocationTrait<M>::Init(x.n_rows(), x.n_cols(), z);
          SubExpert<PrecisionType>(x.length(), x.ptr(), y.ptr(), z->ptr());
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
     *   template<typename PrecisionType>
     *   void TransposeSquare(Matrix<PrecisionType, false> *X);
     *   // example
     *   fl::la::Matrix<float> x;
     *   fl::la::Random(4, 4, &x);
     *   fl::la::TransposeSquare(&x);
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, Automatically
     *                   deduced by the function arguments
     */
    template<typename PrecisionType>
    static inline void TransposeSquare(Matrix<PrecisionType, false> *X) {
      DEBUG_MATSQUARE(*X);
      index_t nr = X->n_rows();
      for (index_t r = 1; r < nr; r++) {
        for (index_t c = 0; c < r; c++) {
          PrecisionType temp = X->get(r, c);
          X->set(r, c, X->get(c, r));
          X->set(c, r, temp);
        }
      }
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
     *     template<typename PrecisionType>
     *     Transpose(const Matrix<PrecisionType, false> &X,
     *                     Matrix<PrecisionType, false> *Y);
     *   };
     *   // example
     *   fl::la::Matrix<double> x;
     *   fl::la::Matrix<double> y;
     *   fl::la::Random(4, 6, &x);
     *   fl::la::Transpse<fl::la::init>(x, &y);
     *   // or if we want to initialize on our own
     *   fl::la::Matrix w;
     *   w.Init(6, 4);
     *   fl::la::Transpose<fl::la::Overwrite>(x, &y);
     *
     * @endcode
     * @param PrecisionType, template parameter for the precision, automatically
     *                   deduced from the functtion arguments
     * @param M, it can be fl::la::Init if we want the function to initialize the
     *           result, or fl::la::Overwrite if the result has already been initialized
     */
    template<fl::la::MemoryAlloc M>
    class Transpose {
      public:
        template<typename PrecisionType>
        Transpose(const Matrix<PrecisionType, false> &X,
                  Matrix<PrecisionType, false> *Y) {
          AllocationTrait<M>::Init(X.n_cols(), X.n_rows(), Y);
          index_t nr = X.n_rows();
          index_t nc = X.n_cols();
          for (index_t r = 0; r < nr; r++) {
            for (index_t c = 0; c < nc; c++) {
              Y->set(c, r, X.get(r, c));
            }
          }
        }
    };




    /**
     * @brief Matrix Multiplication, close to the Blas format
     *        (\f$  C = A * B + b * C \f$)
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
     *    template<typename PrecisionType>
     *    MulExpert(PrecisionType alpha,
     *              const Matrix<PrecisionType, false> &a,
     *              const Matrix<PrecisionType, false> &b,
     *              PrecisionType beta,
     *              Matrix<PrecisionType, false> *c);
     *     template<typename PrecisionType>
     *  MulExpert(const PrecisionType alpha,
     *            const Matrix<PrecisionType, false> &A,
     *            const Matrix<PrecisionType, true> &x,
     *                  PrecisionType beta,
     *                  Matrix<PrecisionType, true> *y);
     *  };
     *  // example
     *  fl::la::Matrix<double> a;
     *  fl::la::Matrix<double> b;
     *  fl::la::Matrix<double> c;
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
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float,double. It is automatically deduced from the function arguments
     */
    template < fl::la::TransMode IsTransA, fl::la::TransMode IsTransB = fl::la::NoTrans >
    class MulExpert {
      public:
        template<typename PrecisionType>
        MulExpert(const PrecisionType alpha,
                  const Matrix<PrecisionType, false> &A,
                  const Matrix<PrecisionType, true> &x,
                  PrecisionType beta,
                  Matrix<PrecisionType, true> *y) {
          DEBUG_ASSERT(x.ptr() != y->ptr());
          DEBUG_SAME_SIZE(IsTransA?A.n_cols():A.n_rows(), y->n_rows());
          DEBUG_SAME_SIZE(x.n_cols(), y->n_cols());
          if (IsTransA == true) {
            DEBUG_SAME_SIZE(A.n_rows(), x.n_rows());
            CppBlas<PrecisionType>::gemv("T", A.n_rows(), A.n_cols(),
                                         alpha, A.ptr(), A.n_rows(), x.ptr(), 1,
                                         beta, y->ptr(), 1);
          }
          else {
            DEBUG_SAME_SIZE(A.n_cols(), x.n_rows());
            CppBlas<PrecisionType>::gemv("N", A.n_rows(), A.n_cols(),
                                         alpha, A.ptr(), A.n_rows(), x.ptr(), 1,
                                         beta, y->ptr(), 1);
          }
        }

        template<typename PrecisionType>
        MulExpert(PrecisionType alpha,
                  const Matrix<PrecisionType, false> &A,
                  const Matrix<PrecisionType, false> &B,
                  PrecisionType beta,
                  Matrix<PrecisionType, false> *C) {
          DEBUG_ASSERT(B.ptr() != C->ptr());
          if (IsTransB == true) {
            if (IsTransA == true) {
              DEBUG_SAME_SIZE(A.n_rows(), B.n_cols());
              DEBUG_SAME_SIZE(A.n_cols(), C->n_rows());
              DEBUG_SAME_SIZE(C->n_cols(), B.n_rows());
              CppBlas<PrecisionType>::gemm("T", "T",
                                           C->n_rows(), C->n_cols(),  A.n_rows(),
                                           alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
                                           beta, C->ptr(), C->n_rows());
            }
            else {
              DEBUG_SAME_SIZE(A.n_cols(), B.n_cols());
              DEBUG_SAME_SIZE(A.n_rows(), C->n_rows());
              DEBUG_SAME_SIZE(C->n_cols(), B.n_rows());
              CppBlas<PrecisionType>::gemm("N", "T",
                                           C->n_rows(), C->n_cols(),  A.n_cols(),
                                           alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
                                           beta, C->ptr(), C->n_rows());
            }
          }
          else {
            if (IsTransA == true) {
              DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
              DEBUG_SAME_SIZE(A.n_cols(), C->n_rows());
              DEBUG_SAME_SIZE(C->n_cols(), B.n_cols());
              CppBlas<PrecisionType>::gemm("T", "N",
                                           C->n_rows(), C->n_cols(),  A.n_rows(),
                                           alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
                                           beta, C->ptr(), C->n_rows());
            }
            else {
              DEBUG_SAME_SIZE(A.n_cols(), B.n_rows());
              DEBUG_SAME_SIZE(A.n_rows(), C->n_rows());
              DEBUG_SAME_SIZE(C->n_cols(), B.n_cols());
              CppBlas<PrecisionType>::gemm("N", "N",
                                           C->n_rows(), C->n_cols(),  A.n_cols(),
                                           alpha, A.ptr(), A.n_rows(), B.ptr(), B.n_rows(),
                                           beta, C->ptr(), C->n_rows());
            }
          }
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
     *    template<typename PrecisionType,
     *             bool IsVectorBool>
     *    Mul(const Matrix<PrecisionType, false> &a,
     *        const Matrix<PrecisionType, IsVectorBool> &b,
     *        Matrix<PrecisionType, IsVectorBool> *c);
     *   };
     *   // example
     *   fl::la::Matrix<double> a;
     *   fl::la::Matrix<double> b;
     *   fl::la::Matrix<double> c;
     *   fl::la::Random(4, 3, &a);
     *   fl::la::Random(3, 4, &b);
     *   Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(a, b, &c);
     * @endcode
     *
     * @param M, It can be fl::la::Init, or fl::la::Overwrite, depending on whether
     *           we want the result to be initialized by Mul or not.
     * param IsTransA, It can be fl::la::NoTrans or fl::la::Trans, depending on
     *                 whether we want to use A's transpose or not.
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced by the funtion arguments
     * @param IsVectorBool, bool parameter for backward compatibility,
     *                  automatically deduced from the function arguments
     *
     */
    template < fl::la::MemoryAlloc M,
    fl::la::TransMode IsTransA = fl::la::NoTrans,
    fl::la::TransMode IsTransB = fl::la::NoTrans >
    class Mul {
      public:
        template < typename PrecisionType,
        bool IsVectorBool >
        Mul(const Matrix<PrecisionType, false> &A,
            const Matrix<PrecisionType, IsVectorBool> &B,
            Matrix<PrecisionType, IsVectorBool> *C) {
          index_t n;
          if (IsTransA == true) {
            n = A.n_cols();
          }
          else {
            n = A.n_rows();
          }
          AllocationTrait<M>::Init(n, B.n_cols(), C);
          MulExpert<IsTransA>(
            PrecisionType(1.0), A, B, PrecisionType(0.0), C);
        }
        template<typename PrecisionType>
        Mul(const Matrix<PrecisionType, false> &A,
            const Matrix<PrecisionType, false> &B,
            Matrix<PrecisionType, false> *C) {
          index_t nr = 0;
          index_t nc = 0;
          if (IsTransA == true) {
            nr = A.n_cols();
          }
          else {
            nr = A.n_rows();
          }
          if (IsTransB == true) {
            nc = B.n_rows();
          }
          else {
            nc = B.n_cols();
          }

          AllocationTrait<M>::Init(nr, nc, C);
          MulExpert<IsTransA, IsTransB>(
            PrecisionType(1.0), A, B, PrecisionType(0.0), C);
        }
    };




    /* --- Wrappers for LAPACK --- */
    /**
     * @brief Destructively computes an LU decomposition of a matrix.
     *
     * Stores L and U in the same matrix (the unitary diagonal of L is
     * implicit).
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double
     *
     * @param pivots a size min(M, N) array to store pivotes
     * @param A_in_LU_out an M-by-N matrix to be decomposed; overwritten
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static inline void PLUExpert(f77_integer *pivots,
                                 Matrix<PrecisionType, false> *A_in_LU_out,
                                 success_t *success) {
      f77_integer info;
      CppLapack<PrecisionType>::getrf(A_in_LU_out->n_rows(),
                                      A_in_LU_out->n_cols(),
                                      A_in_LU_out->ptr(), A_in_LU_out->n_rows(), pivots, &info);
      *success = SUCCESS_FROM_LAPACK(info);
    }
    /**
     * Pivoted LU decomposition of a matrix.
     * @code
     *   // We use PLU as a class constructor because M has to be explicitly
     *   // defined, while PrecisionType is automatically deduced by the
     *   // function arguments. The following example will shed some light
     *   template<tMemoryAlloc M>
     *   class PLU {
     *   public:
     *    template<MemoryAlloc M>
     *     PLU(const Matrix<PrecisionType, false> &A,
     *         std::vector<f77_integer> *pivots, Matrix<PrecisionType, false> *L,
     *         Matrix<PrecisionType, false> *U);
     *     success_t success;
     *   };
     *   fl::la::Matrix<double> a;
     *   fl::la::Random(5, 5, &a);
     *   fl::la::std::vector<double> pivots;
     *   fl::la::Matrix<double> l;
     *   fl::la::Matrix<double> u;
     *   success_t success =
     *     fl::la::PLU<fl::la::Init>(a, &pivots, &l, &u).success;
     *   // The strange syntax ().success in the end was necessary
     *   // since the constructors don't return anything. If you
     *   // just want to do PLU without carring about the success
     *   // you don't have to put the ()..success in the end
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced by the function arguments
     * @param M, it can be fl::la::Init if PLU will initialize the result or
     *           fl::la::Overwrite if the soace has already been allocated
     * @param A the matrix to be decomposed
     * @param pivots a fresh array to be initialized to length min(M,N)
     *        and filled with the permutation pivots
     * @param L a fresh matrix to be initialized to size M-by-min(M,N)
     *        and filled with the lower triangular matrix
     * @param U a fresh matrix to be initialized to size min(M,N)-by-N
     *        and filled with the upper triangular matrix
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M>
    class PLU {
      public:
        template<typename PrecisionType>
        PLU(const Matrix<PrecisionType, false> &A,
            std::vector<f77_integer> *pivots, Matrix<PrecisionType, false> *L,
            Matrix<PrecisionType, false> *U, success_t *success) {
          index_t m = A.n_rows();
          index_t n = A.n_cols();

          if (m > n) {
            pivots->resize(n);
            L->Copy(A);
            AllocationTrait<M>::Init(n, n, U);
            PLUExpert(pivots->begin(), L, success);

            if (!PASSED(*success)) {
              return;
            }

            for (index_t j = 0; j < n; j++) {
              PrecisionType *lcol = L->GetColumnPtr(j);
              PrecisionType *ucol = U->GetColumnPtr(j);
              ::memcpy(ucol, lcol, (j + 1) * sizeof(PrecisionType));
              ::memset(ucol + j + 1, 0, (n - j - 1) * sizeof(PrecisionType));
              ::memset(lcol, 0, j * sizeof(PrecisionType));
              lcol[j] = 1.0;
            }
          }
          else {
            pivots->resize(m);
            L->Init(m, m);
            AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), U);
            U->CopyValues(A);
            PLUExpert(pivots->begin(), U, success);

            if (!PASSED(*success)) {
              *success;
            }

            for (index_t j = 0; j < m; j++) {
              PrecisionType *lcol = L->GetColumnPtr(j);
              PrecisionType *ucol = U->GetColumnPtr(j);

              ::memset(lcol, 0, j * sizeof(PrecisionType));
              lcol[j] = 1.0;
              ::memcpy(lcol + j + 1, ucol + j + 1, (m - j - 1) * sizeof(PrecisionType));
              ::memset(ucol + j + 1, 0, (m - j - 1) * sizeof(PrecisionType));
            }
          }
          return ;
        }
    };
    /**
     * @brief Destructively computes an inverse from a PLU decomposition.
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double
     *
     * @param pivots the pivots array from PLU decomposition
     * @param LU_in_B_out the LU decomposition; overwritten with inverse
     * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static inline void InverseExpert(f77_integer *pivots,
                                     Matrix<PrecisionType, false> *LU_in_B_out,
                                     success_t *success) {
      f77_integer info;
      f77_integer n = LU_in_B_out->n_rows();
      f77_integer lwork = CppLapack<PrecisionType>::getri_block_size * n;
      boost::scoped_array<PrecisionType> work(new PrecisionType[lwork]);
      DEBUG_MATSQUARE(*LU_in_B_out);
      CppLapack<PrecisionType>::getri(n, LU_in_B_out->ptr(), n, pivots,
                                      work.get(), lwork, &info);
      *success = SUCCESS_FROM_LAPACK(info);
    }
    /**
     * @brief Inverts a matrix in place
     * (\f$A \gets A^{-1}\f$).
     *
     * @code
     *  template<typename PrecisionType>
     *  success_t InverseExpert(Matrix<PrecisionType, false> *A)
     *  // example
     *  fl::la::Matrix a;
     *  a.Init(2, 2);
     *  // assign a to [4 0; 0 0.5]
     *  a[0]=4; a[1]=0; a[2]=0; a[3]=0.5
     *  fl::la::Inverse(&a);
     *  //a is now [0.25 0; 0 2.0]
     * @endcode
      * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     *
     * @param A an N-by-N matrix to invert
     * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static void InverseExpert(Matrix<PrecisionType, false> *A, success_t *success) {
      boost::scoped_array<f77_integer> pivots(new f77_integer[A->n_rows()]);

      PLUExpert(pivots.get(), A, success);

      if (!PASSED(*success)) {
        return;
      }

      InverseExpert(pivots.get(), A, success);
    }


    /**
     * @brief Set a matrix to the inverse of another matrix
     *        (\f$B \gets A^{-1}\f$).
     *
     * @code
     *   // Inverse is a function but we call it as a class constructor
     *   // This is because M has to be explicitly defined, while Precision
     *   // can be deduced by input. Since constructors don't return values
     *   // we store the success of the function in success member variable
     *   // The following result will show you how to use .
     *   template<typename PrecisionType, MemoryAlloc M>
     *    class {
     *     public:
     *      template<typename PrecisionType>
     *      Inverse(const Matrix<PrecisionType, false> &A,
     *                    Matrix<PrecisionType, false> *B);
     *      success_t success;
     *    };
     *    //example
     *    fl::la::Matrix a;
     *    a.Init(2, 2);
     *    // assign a to [4 0; 0 0.5]
     *    a[0]=4; a[1]=0; a[2]=0; a[3]=0.5;
     *    fl::la::Matrix<double> b;
     *    success_t success = fl::la::Inverse<fl::la::Init>(a, &b).success;
     *    //b is now [0.25 0; 0 2.0]
     *    // Another example showing the usage of fl::la::Overwrite
     *    fl::la::Matrix<double> c;
     *    c.Init(2, 2);
     *    success = fl::la::Inverse<fl::la::Overwrite>(a, &c);
     *   // c is now [0.25 0; 0 2.0]
     * @endcode
     *
     * @param M, It can be fl::la::Init if the result will be initialized by
     *           the function, or fl::la::Overwrite, if the result is initialized
     *           externally and the function just overwrites it
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced by the function arguments
     *
     * @param A an N-by-N matrix to invert
     * @param B an N-by-N matrix to store the results
     * @return SUCCESS_PASS if invertible, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M>
    class Inverse {
      public:
        template<typename PrecisionType>
        Inverse(const Matrix<PrecisionType, false> &A,
                Matrix<PrecisionType, false> *B,
                success_t *success) {
          boost::scoped_array<f77_integer> pivots(new f77_integer[A.n_rows()]);
          AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), B);
          if (likely(A.ptr() != B->ptr())) {
            B->CopyValues(A);
          }
          PLUExpert(pivots.get(), B, success);

          if (!PASSED(*success)) {
            return ;
          }
          InverseExpert(pivots.get(), B, success);
        }
    };

    /**
     * @bried Returns the determinant of a matrix
     *       (\f$\det A\f$).
     *
     * @code
     *   template<typename PrecisionType>
     *   long double Determinant(const Matrix<PrecisionType, false> &A);
     *   // example
     *   fl::la::Matrix a;
     *   a.Init(2, 2);
     *   // assign a to [4 0; 0 0.5]
     *   a[0]=4; a[1]=0; a[2]=0; a[3]=0.5;
     *   double det = fl::la::Determinant(a);
     * // ... det is equal to 2.0
     * @endcode
      * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     *
     * @param A the matrix to find the determinant of
     * @return the determinant; note long double for large exponents
     */
    template<typename PrecisionType>
    static long double Determinant(const Matrix<PrecisionType, false> &A) {
      DEBUG_MATSQUARE(A);
      int n = A.n_rows();
      boost::scoped_array<f77_integer> pivots(new f77_integer[n]);
      Matrix<PrecisionType, false> LU;

      LU.Copy(A);
      success_t success;
      PLUExpert<PrecisionType>(pivots.get(), &LU, &success);

      long double det = 1.0;

      for (index_t i = 0; i < n; i++) {
        if (pivots[i] != i + 1) {
          // pivoting occured (note FORTRAN has 1-based indexing)
          det = -det;
        }
        det *= LU.get(i, i);
      }

      return det;
    }

    /**
     * @brief Returns the log-determinant of a matrix
     *        (\f$\ln |\det A|\f$).
     *
     * This is effectively log(fabs(Determinant(A))).
     * @code
     *   template<typename PrecisionType>
     *   PrecisionType DeterminantLog(const Matrix<PrecisionType, false> &A,
     *                            int *sign_out);
     *   // example
     *   fl::la::Matrix a;
     *   a.Init(2, 2);
     *   // assign a to [4 0; 0 0.5]
     *   a[0]=4; a[1]=0; a[2]=0; a[3]=0.5;
     *   double det = fl::la::DeterminantLog(a, NULL);
     *   int sign;
     *   det = fl::la::DeterminantLog(a, &sign);
     *   // det is equal to log(2.0)
     *   // sign is 1;
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced by the function arguments
     *
     * @param A the matrix to find the determinant of
     * @param sign_out set to -1, 1, or 0; pass NULL to disable
     * @return the log of the determinant or NaN if A is singular
     */
    template<typename PrecisionType>
    static PrecisionType DeterminantLog(const Matrix<PrecisionType, false> &A, int *sign_out) {
      DEBUG_MATSQUARE(A);
      int n = A.n_rows();
      boost::scoped_array<f77_integer> pivots(new f77_integer[n]);
      Matrix<PrecisionType, false> LU;

      LU.Copy(A);
      success_t success;
      PLUExpert<PrecisionType>(pivots.get(), &LU, &success);

      PrecisionType log_det = 0.0;
      int sign_det = 1;

      for (index_t i = 0; i < n; i++) {
        if (pivots[i] != i + 1) {
          // pivoting occured (note FORTRAN has one-based indexing)
          sign_det = -sign_det;
        }

        PrecisionType value = LU.get(i, i);

        if (value < 0) {
          sign_det = -sign_det;
          value = -value;
        }
        else
          if (!(value > 0)) {
            sign_det = 0;
            log_det = std::numeric_limits<PrecisionType>::quiet_NaN();
            break;
          }

        log_det += log(value);
      }

      if (sign_out) {
        *sign_out = sign_det;
      }

      return log_det;
    }

    /**
     * @brief Destructively solves a system of linear equations (X st A * X = B).
     *
     * This computes the PLU factorization in A as a side effect, but
     * you are free to ignore it.
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced by the function arguments
     *
     * @param pivots a size N array to store pivots of LU decomposition
     * @param A_in_LU_out an N-by-N matrix multiplied by x; overwritten
     *        with its LU decomposition
     * @param k the number of columns on the right-hand side
     * @param B_in_X_out an N-by-K matrix ptr of desired products;
     *        overwritten with solutions (must not alias A_in_LU_out)
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static inline void SolveExpert(
      f77_integer *pivots, Matrix<PrecisionType, false> *A_in_LU_out,
      index_t k, PrecisionType *B_in_X_out,
      success_t *success) {
      DEBUG_MATSQUARE(*A_in_LU_out);
      f77_integer info;
      f77_integer n = A_in_LU_out->n_rows();
      CppLapack<PrecisionType>::gesv(n, k, A_in_LU_out->ptr(), n, pivots,
                                     B_in_X_out, n, &info);
      *success = SUCCESS_FROM_LAPACK(info);
    }
    /**
     * @brief Inits a matrix to the solution of a system of linear equations
     *        (X st A * X = B).
     *
     * @code
     *   // Although Solve is a function it is being used as a function constructor
     *   // because M has to be explicitly defined, but the other templated arguments
     *   // can be deduced by the function arguments. The following example will
     *   // shed some light
     *   template<MemoryAlloc M>
     *   class Solve {
     *    public:
     *     template<typename PrecisionType, bool IsVectorBool>
     *     Solve(const Matrix<PrecisionType, false> &A,
     *           const Matrix<PrecisionType, IsVectorBool> &B,
     *                 Matrix<PrecisionType, IsVectorBool> *X);
     *     success_t success;
     *   };
     *   // example
     *   fl::la::Matrix<double> a;
     *   a.Init(2, 2);
     *   fl::la::Matrix<double> b;
     *   b.Init(2, 2);
     *   // assign A to [1 3; 2 10]
     *   a[0]=1; a[1]=2; a[2]=3; a[3]=10;
     *   // assign B to [2 3; 8 10]
     *   b[0]=2; b[1]=3; b[2]=8; b[3]=10;
     *   fl::la::Matrix<double> x; // Not initialized
     *   success_t success = fl::la::Solve<fl::la::Init>(a, b, &x).success;
     *   // x is now [-1 0; 1 1]
     *   fl::la::Matrix<double> c;
     *   fl::la::Mul<fl::la::Init>(a, x, &c);
     *   // b and c should be equal (but for round-off)
     * @endcode
     *
     * @param M, it can be fl::la::Init, if the function will initialize the
     *           result, or fl::la::Overwrite if the result has already been
     *           initialized and it is going to be overwritten
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It can be deduced from the function arguments
     * @param A an N-by-N matrix multiplied by x
     * @param B a size N-by-K matrix of desired products
     * @param X a fresh matrix to be initialized to size N-by-K
     *        and filled with solutions
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M>
    class Solve {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        Solve(const Matrix<PrecisionType, false> &A,
              const Matrix<PrecisionType, IsVectorBool> &B,
              Matrix<PrecisionType, IsVectorBool> *X,
              success_t *success) {
          DEBUG_MATSQUARE(A);
          DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
          Matrix<PrecisionType, false> tmp;
          index_t n = B.n_rows();
          boost::scoped_array<f77_integer> pivots(new f77_integer[n]);
          tmp.Copy(A);
          AllocationTrait<M>::Init(B.n_rows(), B.n_cols(), X);
          X->CopyValues(B);
          SolveExpert<PrecisionType>(pivots.get(), &tmp, B.n_cols(), X->ptr(), success);
        }
    };

    /**
     * @brief Inits a matrix to the solution of a system of linear equations
     *        (X st A * X = B).
     *  WHERE A is Triangular
     * @code
     *   // Although Solve is a function it is being used as a function constructor
     *   // because M has to be explicitly defined, but the other templated arguments
     *   // can be deduced by the function arguments. The following example will
     *   // shed some light
     *   template<MemoryAlloc M, TransMode IsTransA>
     *   class SolveTriangular {
     *    public:
     *     template<typename PrecisionType, bool IsVectorBool>
     *     SolveTriangular(const Matrix<PrecisionType, false> &A,
     *                     bool islower,
     *                     const Matrix<PrecisionType, IsVectorBool> &B,
     *                     Matrix<PrecisionType, IsVectorBool> *X);
     *     success_t success;
     *   };
     *   // example
     *   fl::la::Matrix<double> a;
     *   a.Init(2, 2);
     *   fl::la::Matrix<double> b;
     *   b.Init(2, 2);
     *   // assign A to [1 0; 2 10]
     *   a[0]=1; a[1]=0; a[2]=2; a[3]=10;
     *   // assign B to [2 3; 8 10]
     *   b[0]=2; b[1]=3; b[2]=8; b[3]=10;
     *   fl::la::Matrix<double> x; // Not initialized
     *   success_t success = fl::la::SolveTriangular<fl::la::Init, fl::la::NoTrans>(a, true, b, &x).success;
     *   
     * @endcode
     *
     * @param M, it can be fl::la::Init, if the function will initialize the
     *           result, or fl::la::Overwrite if the result has already been
     *           initialized and it is going to be overwritten
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It can be deduced from the function arguments
     * @param A an N-by-N matrix multiplied by x
     * @param islower bool indicating if it is lower or upper triangular
     * @param B a size N-by-K matrix of desired products
     * @param X a fresh matrix to be initialized to size N-by-K
     *        and filled with solutions
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M, fl::la::TransMode IsTransA>
    class SolveTriangular {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        SolveTriangular(const Matrix<PrecisionType, false> &A,
              bool islower,
              const Matrix<PrecisionType, IsVectorBool> &B,
              Matrix<PrecisionType, IsVectorBool> *X,
              success_t *success) {
          DEBUG_MATSQUARE(A);
          DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
          Matrix<PrecisionType, false> tmp;
          tmp.Copy(A);
          AllocationTrait<M>::Init(B.n_rows(), B.n_cols(), X);
          X->CopyValues(B);
          f77_integer info;
          CppLapack<PrecisionType>::trtrs(
              islower?"L":"U",
              IsTransA?"T":"N",
              "N",
              (const f77_integer)A.n_rows(),
              (const f77_integer)B.n_cols(),
              tmp.ptr(),
              (const f77_integer)A.n_cols(), 
              X->ptr(), 
              (const f77_integer)B.n_rows(), 
              &info);
          *success = SUCCESS_FROM_LAPACK(info);
          if (info<0) {
            fl::logger->Warning()<<"The "<<-info<<"th element has an invalid value"
              <<std::endl;
          } else {
            if (info>0) {
              fl::logger->Warning()<<"The "<<info<<"th element of the diagonal is zero"
               <<std::endl; 
            }
          }
        }
    };


    /**
     * @brief Destructively performs a QR decomposition (A = Q * R).
     *        Factorizes a matrix as a rotation matrix (Q) times a reflection
     *        matrix (R); generalized for rectangular matrices.
     * @code
     *   template<typename PrecisionType>
     *   success_t QRExpert(Matrix<PrecisionType, false> *A_in_Q_out,
     *                      Matrix<PrecisionType, false>  *R);
     *   // example
     *   // This is matrix a, but after QR decomposition, it will store q
     *   fl::la::Matrix<double> a_in_q_out;
     *   fl::la::Random(4, 4, &a_in_q_out);
     *   fl::la::Matrix<double> r;
     *   // it must be initialized;
     *   r.Init(4, 4);
     *   fl::la::QRExpert(&a_in_q_out, &r);
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It can be deduced by the function arguments
     * @param A_in_Q_out an M-by-N matrix to factorize; overwritten with
     *        Q, an M-by-min(M,N) matrix (remaining columns are garbage,
     *        but are not removed from the matrix)
     * @param R a min(M,N)-by-N matrix to store results (must not be
     *        A_in_Q_out)
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static void QRExpert(Matrix<PrecisionType, false> *A_in_Q_out,
                         Matrix<PrecisionType, false>  *R, success_t *success) {
      f77_integer info;
      f77_integer m = A_in_Q_out->n_rows();
      f77_integer n = A_in_Q_out->n_cols();
      f77_integer k = std::min(m, n);
      f77_integer lwork = n * CppLapack<PrecisionType>::geqrf_dorgqr_block_size;
      boost::scoped_array<PrecisionType> tau(new PrecisionType[k + lwork]);
      PrecisionType *work = tau.get() + k;

      // Obtain both Q and R in A_in_Q_out
      CppLapack<PrecisionType>::geqrf(m, n, A_in_Q_out->ptr(), m,
                                      tau.get(), work, lwork, &info);

      if (info != 0) {
        *success = SUCCESS_FROM_LAPACK(info);
        return;
      }

      // Extract R
      for (index_t j = 0; j < n; j++) {
        PrecisionType *r_col = R->GetColumnPtr(j);
        PrecisionType *q_col = A_in_Q_out->GetColumnPtr(j);
        int i = std::min(j + 1, index_t(k));
        ::memcpy(r_col, q_col, i * sizeof(PrecisionType));
        ::memset(r_col + i, 0, (k - i) * sizeof(PrecisionType));
      }

      // Fix Q
      CppLapack<PrecisionType>::orgqr(m, k, k, A_in_Q_out->ptr(), m,
                                      tau.get(), work, lwork, &info);

      *success = SUCCESS_FROM_LAPACK(info);

    }
    /**
     * @brief Init matrices to a QR decomposition (A = Q * R).
     *         Factorizes a matrix as a rotation matrix (Q) times a reflection
     *         matrix (R); generalized for rectangular matrices.
     *
     * @code
     *   // Although QR is a function we are using it as a class constructor. This
     *   // is becauce M has to be explicitly defined, while other template
     *   // arguments can be deduced from the function arguments. The following
     *   // result will make it more clear
     *   template<MemoryAlloc M>
     *   class QR {
     *    public:
     *     template<typename PrecisionType>
     *     QR(const Matrix<PrecisionType, false> &A,
     *              Matrix<PrecisionType, false> *Q,
     *              Matrix<PrecisionType, false> *R);
     *     success_t success;
     *   };
     *   fl::la::Matrix<double> a;
     *   // assign A to [3 5; 4 12]
     *   a.Init(2,2);
     *   a[0]=3; a[1]=4; a[2]=5; a[3]=12;
     *   fl::la::Matrix<double> q;
     *   fl::la::Matrix<double> r;
     *   success_t success = fl::la::QR<fl::la::Init>(a, &q, &r).success;
     *   // q is now [-0.6 -0.8; -0.8 0.6]
     *   // r is now [-5.0 -12.6; 0.0 3.2]
     *   fl::la::Matrix<double> b;
     *   fl::la::Mul<fl::la::Init>(q, r, &b)
     *   // a and b should be equal (but for round-off)
     * @endcode
      * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     *
     * @param A an M-by-N matrix to factorize
     * @param Q a fresh matrix to be initialized to size M-by-min(M,N)
              and filled with the rotation matrix
     * @param R a fresh matrix to be initialized to size min(M,N)-by-N
              and filled with the reflection matrix
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M,
    fl::la::TransMode IsTransA = fl::la::NoTrans>
    class QR {
      public:
        template<typename PrecisionType>
        QR(const Matrix<PrecisionType, false> &A,
           Matrix<PrecisionType, false> *Q,
           Matrix<PrecisionType, false> *R,
           success_t *success) {
          index_t k = std::min(A.n_rows(), A.n_cols());
          if (IsTransA==fl::la::NoTrans) {
            AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), Q);
            Q->CopyValues(A);
            AllocationTrait<M>::Init(k, A.n_cols(), R);
            R->SetAll(0);
            QRExpert(Q, R, success);
            Q->ResizeNoalias(k);
          }
          if (IsTransA==fl::la::Trans) {
             Matrix<PrecisionType, false> Qaux;
             AllocationTrait<fl::la::Init>::Init(A.n_cols(), 
                 A.n_rows(), &Qaux);
             for(index_t i=0; i<A.n_rows(); ++i) {
               for(index_t j=0; j<A.n_cols(); ++j) {
                 Qaux.set(j, i, A.get(i, j));
               }
             }
             AllocationTrait<M>::Init(k, A.n_rows(), R);      
             R->SetAll(0);
             QRExpert(&Qaux, R, success);
             AllocationTrait<M>::Init(k, A.n_cols(), Q);              
             for(index_t i=0; i<Qaux.n_rows(); ++i) {
               for(index_t j=0; j<k; ++j) {
                 Q->set(j,i, Qaux.get(i, j));
               }
             }
          }
        }
    };
    /**
     * @brief Destructive Schur decomposition (A = Z * T * Z').
     *        This uses DGEES to find a Schur decomposition, but is also the best
     *         way to find just eigenvalues.
     * @code
     *   template<typename PrecisionType>
     *   success_t SchurExpert(Matrix<PrecisionType, false> *A_in_T_out,
     *   PrecisionType *w_real, PrecisionType *w_imag, PrecisionType *Z);
     *   // example
     *
     *   // Here we store the initial matrix A, but after decomposition
     *   // T matrix will be stored there
     *   fl::la::Matrix<double> a_in_t_out;
     *   fl::la::Random(3, 3, &a_in_t_out);
     *   double w_real[3];
     *   double w_imag[3];
     *   double z[3][3];
     *   fl::la::ShurExpert(&a_in_t_out, w_real, w_imag, z);
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is deduced by the function arguments
     *
     * @param A_in_T_out am N-by-N matrix to decompose; overwritten
     *        with the Schur form
     * @param w_real a length-N array to store real eigenvalue components
     * @param w_imag a length-N array to store imaginary components
     * @param Z an N-by-N matrix ptr to store the Schur vectors, or NULL
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static void SchurExpert(Matrix<PrecisionType, false> *A_in_T_out,
                            PrecisionType *w_real, PrecisionType *w_imag, PrecisionType *Z, success_t *success) {
      DEBUG_MATSQUARE(*A_in_T_out);
      f77_integer info;
      f77_integer n = A_in_T_out->n_rows();
      f77_integer sdim;
      const char *job = Z ? "V" : "N";
      PrecisionType d; // for querying optimal work size

      CppLapack<PrecisionType>::gees(job, "N", NULL,
                                     n, A_in_T_out->ptr(), n, &sdim, w_real, w_imag,
                                     Z, n, &d, -1, NULL, &info);
      {
        f77_integer lwork = (f77_integer)d;
        boost::scoped_array<PrecisionType> work(new PrecisionType[lwork]);

        CppLapack<PrecisionType>::gees(job, "N", NULL,
                                       n, A_in_T_out->ptr(), n, &sdim, w_real, w_imag,
                                       Z, n, work.get(), lwork, NULL, &info);
      }

      *success = SUCCESS_FROM_LAPACK(info);
    }

    /**
     * @brief Init matrices to a Schur decompoosition (A = Z * T * Z').
     * @code
     *   // Although Shur is a function, we use it as a class constructor,
     *   // because some of the template parameters have to be explicitly defined
     *   // while others can be deduced from context. The following example
     *   // will make it clear
     *
     *   template<MemoryAlloc M>
     *   class Schur {
     *    public:
     *     template<typename PrecisionType, bool IsVectorBool>
     *      Schur(const Matrix<PrecisionType, false> &A,
     *                  Matrix<PrecisionType, IsVectorBool> *w_real,
     *                  Matrix<PrecisionType, IsVectorBool> *w_imag,
     *                  Matrix<PrecisionType, false> *T,
     *                  Matrix<PrecisionType, false> *Z);
     *      success_t success;
     *    };
     *    // example
     *    fl::la::Matrix<double> a;
     *    fl::la::Random(4, 4, &a);
     *    fl::la::Matrix<double, true> w_real;
     *    fl::la::Matrix<double, true> w_imag;
     *    fl::la::Matrix<double> t;
     *    fl::la::Matrix<double> z;
     *    success_t success = fl::la::Schur<fl::la::Init>(a,
     *        &w_real, &w_imag, &t, &z);
     * @endcode
     * @param M, if it is fl::la::Init, then the function initializes the results
     *           if it is fl::la::Overwrite, then the function just overwrites
     *           the already allocated results
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is deduced by the function arguments
     * @param IsVectorBool, boolean template parameter for backward compatibility
     *                  with Vector. It is deduced by the function arguments
     * @param A an N-by-N matrix to decompose
     * @param w_real a fresh vector to be initialized to length N
     *        and filled with the real eigenvalue components
     * @param w_imag a fresh vector to be initialized to length N
     *        and filled with the imaginary components
     * @param T a fresh matrix to be initialized to size N-by-N
     *        and filled with the Schur form
     * @param Z a fresh matrix to be initialized to size N-by-N
     *        and filled with the Schur vectors
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M>
    class Schur {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        Schur(const Matrix<PrecisionType, false> &A,
              Matrix<PrecisionType, IsVectorBool> *w_real,
              Matrix<PrecisionType, IsVectorBool> *w_imag,
              Matrix<PrecisionType, false> *T,
              Matrix<PrecisionType, false> *Z,
              success_t *success) {
          index_t n = A.n_rows();
          AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), T);
          T->CopyValues(A);
          AllocationTrait<M>::Init(n, 1, w_real);
          AllocationTrait<M>::Init(n, 1, w_imag);
          // w_real->Init(n);
          // w_imag->Init(n);
          AllocationTrait<M>::Init(n, n, Z);
          SchurExpert(T, w_real->ptr(), w_imag->ptr(), Z->ptr(), success);
        }
    };
    /**
     * @brief Destructive, unprocessed eigenvalue/vector decomposition.
     *      Real eigenvectors are stored in the columns of V, while imaginary
     *      eigenvectors occupy adjacent columns of V with conjugate pairs
     *      given by V(:,j) + i*V(:,j+1) and V(:,j) - i*V(:,j+1).
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It can be deduced by the function arguments
     * @param A_garbage an N-by-N matrix to be decomposed; overwritten
     *        with garbage
     * @param w_real a length-N array to store real eigenvalue components
     * @param w_imag a length-N array to store imaginary components
     * @param V_raw an N-by-N matrix ptr to store eigenvectors, or NULL
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static void EigenExpert(Matrix<PrecisionType, false> *A_garbage,
                            PrecisionType *w_real, PrecisionType *w_imag, PrecisionType *V_raw,
                            success_t *success) {
      DEBUG_MATSQUARE(*A_garbage);
      f77_integer info;
      f77_integer n = A_garbage->n_rows();
      const char *job = V_raw ? "V" : "N";
      PrecisionType d; // for querying optimal work size

      CppLapack<PrecisionType>::geev("N", job, n, A_garbage->ptr(), n,
                                     w_real, w_imag, NULL, 1, V_raw, n, &d, -1, &info);
      {
        f77_integer lwork = (f77_integer)d;
        boost::scoped_array<PrecisionType> work(new PrecisionType[lwork]);

        CppLapack<PrecisionType>::geev("N", job, n, A_garbage->ptr(), n,
                                       w_real, w_imag, NULL, 1, V_raw, n,
                                       work.get(), lwork, &info);
      }

      *success =  SUCCESS_FROM_LAPACK(info);
    }

    /**
     * @brief Finds the eigenvalues of a matrix.
     * @code
     *   // Although Eigenvalues is a function we use it as a clss constructor
     *   // because some of the template parameters have to be explicitly deifned
     *   // while others can be deduced. The following example will make it clear
     *
     *   template<MemoryAlloc M>
     *   class Eigenvalues {
     *    public:
     *     // returns imaginary and real eigenvalues
     *     template<typename PrecisionType, bool IsVectorBool>
     *     Eigenvalues(const Matrix<PrecisionType, false> &A,
     *                       Matrix<PrecisionType, IsVectorBool> *w_real,
     *                       Matrix<PrecisionType, IsVectorBool> *w_imag);
     *     // returns only real eigenvalues
     *     template<typename PrecisionType, bool IsVectorBool>
     *     Eigenvalues(const Matrix<PrecisionType, false> &A,
     *                       Matrix<PrecisionType, IsVectorBool> *w);
     *     success_t success;
     *   };
     *
     *  // example
     *  fl::la::Matrix<double> a;
     *  fl::la::Random(6, 6, &a);
     *  fl::la::Matrix<double> w_real;
     *  fl::la::Matrix<double> w_imag;
     *  success_t success=fl::la::Eigenvalues<fl::la::Init>(a,
     *                                &w_real, &w_imag).success;
     *  // we can just get the real eigenvalues
     *  // notice now that we use fl::la::Overwrite since w_real
     *  // has been initialized by the previous call of Eigenvalues
     *  success = fl::la::Eigenvalues<fl::la::Overwrite>(a, &w_real);
     * @endcode
     * @param M, if it is fl::la::Init, then the function initializes the results
     *           if it is fl::la::Overwrite, then the function just overwrites
     *           the already allocated results
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     * @param A an N-by-N matrix to find eigenvalues for
     * @param w_real a fresh vector to be initialized to length N
     *        and filled with the real eigenvalue components
     * @param w_imag a fresh vector to be initialized to length N
     *        and filled with the imaginary components
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M>
    class Eigenvalues {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        Eigenvalues(const Matrix<PrecisionType, false> &A,
                    Matrix<PrecisionType, IsVectorBool> *w_real,
                    Matrix<PrecisionType, IsVectorBool> *w_imag,
                    success_t *success) {
          DEBUG_MATSQUARE(A);
          int n = A.n_rows();
          AllocationTrait<M>::Init(n, 1, w_real);
          AllocationTrait<M>::Init(n, 1, w_imag);
          Matrix<PrecisionType, false> tmp;
          tmp.Copy(A);
          SchurExpert<PrecisionType>(&tmp, w_real->ptr(), w_imag->ptr(), NULL, success);
        };

        /** @brief The tridiagonal solver.
         */
        template<typename PrecisionType>
        Eigenvalues(const Matrix<PrecisionType, true> &diagonal_elements,
                    const Matrix<PrecisionType, true> &offdiagonal_elements,
                    Matrix<PrecisionType, true> *eigenvalues,
                    success_t *success) {

          DEBUG_ASSERT(diagonal_elements.length() ==
                       offdiagonal_elements.length() + 1);
          int n = diagonal_elements.length();
          f77_integer info;
          boost::scoped_array<PrecisionType> d_tmp(new PrecisionType[n]);
          boost::scoped_array<PrecisionType> e_tmp(new PrecisionType[n - 1]);
          for (int i = 0; i < offdiagonal_elements.length(); i++) {
            d_tmp[i] = diagonal_elements[i];
            e_tmp[i] = offdiagonal_elements[i];
          }
          d_tmp[n - 1] = diagonal_elements[n - 1];

          // Call the tridiagonal eigensolver.
          CppLapack<PrecisionType>::sterf(n, d_tmp.get(), e_tmp.get(), &info);

          // Copy back the results.
          AllocationTrait<M>::Init(n, 1, eigenvalues);
          for (int i = 0; i < n; i++) {
            (*eigenvalues)[i] = d_tmp[i];
          }
          *success = ((info == 0) ? SUCCESS_PASS : SUCCESS_FAIL);
        }

        template<typename PrecisionType, bool IsVectorBool>
        Eigenvalues(const Matrix<PrecisionType, false> &A,
                    Matrix<PrecisionType, IsVectorBool> *w,
                    success_t *success) {
          DEBUG_MATSQUARE(A);
          int n = A.n_rows();
          AllocationTrait<M>::Init(n, 1, w);
          boost::scoped_array<PrecisionType> w_imag(new PrecisionType[n]);

          Matrix<PrecisionType, false> tmp;
          tmp.Copy(A);
          SchurExpert<PrecisionType>(&tmp, w->ptr(), w_imag.get(), NULL, success);

          if (!PASSED(*success)) {
            return ;
          }

          for (index_t j = 0; j < n; j++) {
            if (unlikely(w_imag[j] != 0.0)) {
              (*w)[j] = std::numeric_limits<PrecisionType>::quiet_NaN();
            }
          }
        }
    };



    /**
     * @brief Inits vectors and matrices to the eigenvalues/vectors of a matrix.
     *        Complex eigenvalues/vectors are processed into components rather
     *        than raw conjugate form.
     *
     * @code
     *   // although Eigenvectors is a function, we use it as a class constructor
     *   // because some of the template arguments have to be explicitly set,
     *   // while others are deduced by the contex. The following result will make
     *   // it clear.
     *   template<MemoryAlloc M>
     *   class Eigenvectors {
     *    public:
     *     template<typename PrecisionType, bool IsVectorBool>
     *     Eigenvectors(const Matrix<PrecisionType, false> &A,
     *                        Matrix<PrecisionType, IsVectorBool> *w_real,
     *                        Matrix<PrecisionType, IsVectorBool> *w_imag,
     *                        Matrix<PrecisionType, false> *V_real,
     *                        Matrix<PrecisionType, false> *V_imag);
     *
     *     template<typename PrecisionType, bool IsVectorBool>
     *     Eigenvectors(const Matrix<PrecisionType, false> &A,
     *                        Matrix<PrecisionType, IsVectorBool> *w,
     *                        Matrix<PrecisionType, false> *V);
     *    success_t succes;
     *  };
     *  // example
     *  fl::la::Matrix<double> A;
     *  a.Init(2, 2);
     *  // assign A to [0 1; -1 0]
     *  a[0]=0; a[1]=-1;a[2]=1;a[3]=0;
     *  fl::la::Matrix<double> w_real;
     *  fl::la::Matrix<double> w_imag; // Not initialized
     *  fl::la::Matrix<double> v_real; // Not initialized
     *  fl::la::Matrix<double> v_imag; // Not initialized
     *  success_t success = fl::la::Eigenvectors<fl::la::Init>(a, &w_real,
     *      &w_imag, &v_real, &v_imag);
     *  // w_real and w_imag are [0 0] + i*[1 -1]
     *  // V_real and V_imag are [0.7 0.7; 0 0] + i*[0 0; 0.7 -0.7]
     * @endcode
     *
     * @param M, if it is fl::la::Init, then the function initializes the results
     *           if it is fl::la::Overwrite, then the function just overwrites
     *           the already allocated results
     * @param PrecisionType, template parameter for the precision, currently supports
     *        floati, double. It is automatically deduced by the function arguments
     * @param IsVectorBool, bool template argument for backward compatibility with
     *                  Vector. It is deduced by the function arguments
     * @param A an N-by-N matrix to be decomposed
     * @param w_real a fresh vector to be initialized to length N
     *        and filled with the real eigenvalue components
     * @param w_imag a fresh vector to be initialized to length N
     *        and filled with the imaginary eigenvalue components
     * @param V_real a fresh matrix to be initialized to size N-by-N
     *        and filled with the real eigenvector components
     * @param V_imag a fresh matrix to be initialized to size N-by-N
     *        and filled with the imaginary eigenvector components
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M>
    class Eigenvectors {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        Eigenvectors(const Matrix<PrecisionType, false> &A,
                     Matrix<PrecisionType, IsVectorBool> *w_real,
                     Matrix<PrecisionType, IsVectorBool> *w_imag,
                     Matrix<PrecisionType, false> *V_real,
                     Matrix<PrecisionType, false> *V_imag,
                     success_t *success) {
          DEBUG_MATSQUARE(A);
          index_t n = A.n_rows();
          AllocationTrait<M>::Init(n, w_real);
          AllocationTrait<M>::Init(n, w_imag);
          AllocationTrait<M>::Init(n, n, V_real);
          AllocationTrait<M>::Init(n, n, V_imag);

          Matrix<PrecisionType, false> tmp;
          tmp.Copy(A);
          EigenExpert(&tmp,
                      w_real->ptr(), w_imag->ptr(), V_real->ptr(), success);

          if (!PASSED(*success)) {
            return;
          }

          V_imag->SetZero();
          for (index_t j = 0; j < n; j++) {
            if (unlikely(w_imag->get(j) != 0.0)) {
              PrecisionType *r_cur = V_real->GetColumnPtr(j);
              PrecisionType *r_next = V_real->GetColumnPtr(j + 1);
              PrecisionType *i_cur = V_imag->GetColumnPtr(j);
              PrecisionType *i_next = V_imag->GetColumnPtr(j + 1);

              for (index_t i = 0; i < n; i++) {
                i_next[i] = -(i_cur[i] = r_next[i]);
                r_next[i] = r_cur[i];
              }

              j++; // skip paired column
            }
          }
        }

        template<typename PrecisionType, bool IsVectorBool>
        Eigenvectors(const Matrix<PrecisionType, false> &A,
                     Matrix<PrecisionType, IsVectorBool> *w,
                     Matrix<PrecisionType, false> *V,
                     success_t *success) {
          DEBUG_MATSQUARE(A);
          index_t n = A.n_rows();
          AllocationTrait<M>::Init(n, w);
          boost::scoped_array<PrecisionType> w_imag(new PrecisionType[n]);
          AllocationTrait<M>::Init(n, n, V);

          Matrix<PrecisionType, false> tmp;
          tmp.Copy(A);
          EigenExpert(&tmp, w->ptr(), w_imag.get(), V->ptr(), success);

          if (!PASSED(*success)) {
            return;
          }

          for (index_t j = 0; j < n; j++) {
            if (unlikely(w_imag[j] != 0.0)) {
              (*w)[j] = std::numeric_limits<PrecisionType>::quiet_NaN();
            }
          }
          return;
        }
    };

    /**
     * @brief Generalized eigenvalue/vector decomposition of two symmetric matrices.
     *        Compute all the eigenvalues, and optionally, the eigenvectors
     *        of a real generalized eigenproblem, of the form
     *        A*x=(lambda)*B*x, A*Bx=(lambda)*x, or B*A*x=(lambda)*x, where
     *        A and B are assumed to be symmetric and B is also positive definite.
     * @code
     *   template<typename PrecisionType, bool IsVectorBool>
     *   success_t GenEigenSymmetric(int itype,
     *                               Matrix<PrecisionType, false> *A_eigenvec,
     *                               Matrix<PrecisionType, false> *B_chol,
     *                               Matrix<PrecisionType, IsVectorBool> *w);
     *   // example
     *   fl::la::Matrix<double> a;
     *   fl::la::RandomSymmetric(5, 5, &a);
     *   fl::la::Matrix<double> b;
     *   fl::la::RandomSymmetric(5, 5, &b);
     *   fl::la::Matrix<double> w;
     *   w.Init(5, 1);
     *   fl::la:GenEigenSymmetric(1, &a, &b, &w);
     *
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It can be deduced by the function arguments
     * @param IsVectorBool, bool template parameter, for backward compatibility
     *                  with vectors. It is deduced by context
     * @param itype an integer that specifies the problem type to be solved:
     *        = 1: A*x = (lambda)*B*x
     *        = 2: A*B*x = (lambda)*x
     *        = 3: B*A*x = (lambda)*x
     * @param A_eigenvec an N-by-N matrix to be decomposed; overwritten
     *        with the matrix of eigenvectors
     * @param B_chol an N-by-N matrix to be decomposed; overwritten
     *        with triangular factor U or L from the Cholesky factorization
     *        B=U**T*U or B=L*L**T.
     * @param w a length-N array of eigenvalues in ascending order.
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType, bool IsVectorBool>
    static void GenEigenSymmetric(int itype, Matrix<PrecisionType, false> *A_eigenvec,
                                  Matrix<PrecisionType, false> *B_chol,
                                  Matrix<PrecisionType, IsVectorBool>  *w,
                                  success_t *success) {
      DEBUG_MATSQUARE(*A_eigenvec);
      DEBUG_MATSQUARE(*B_chol);
      DEBUG_ASSERT(A_eigenvec->n_rows() == B_chol->n_rows());
      f77_integer itype_f77 = itype;
      f77_integer info;
      f77_integer n = A_eigenvec->n_rows();
      const char *job = "V"; // Compute eigenvalues and eigenvectors.
      PrecisionType d; // for querying optimal work size

      // Allocate the w vector holding the eigenvalues.
      w->Init(A_eigenvec->n_rows());

      CppLapack<PrecisionType>::sygv(&itype_f77, job, "U", n,
                                     A_eigenvec->ptr(), n, B_chol->ptr(), n,
                                     w->ptr(), &d, -1, &info);
      {
        f77_integer lwork = (f77_integer)d;
        boost::scoped_array<PrecisionType> work(new PrecisionType[lwork]);
        w->SetZero();

        CppLapack<PrecisionType>::sygv(&itype_f77, job, "U", n, A_eigenvec->ptr(),
                                       n, B_chol->ptr(), n, w->ptr(),
                                       work.get(), lwork, &info);
      }

      *success = SUCCESS_FROM_LAPACK(info);

    }

    /**
     * @brief Generalized eigenvalue/vector decomposition of two nonsymmetric matrices.
     *        Compute all the eigenvalues, and optionally, the eigenvectors
     *        of a real generalized eigenproblem, of the form
     *        A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x
     *        (alpha_real(j) + alpha_imag(j)*i)/beta(j), j=1,...,N, will be the
     *        generalized eigenvalues.
     * Note: the quotients alpha_real(j)/beta(j) and alpha_imag(j)/beta(j) may
     * easily over- or underflow, and beta(j) may even be zero.  Thus, the
     * user should avoid naively computing the ratio alpha/beta.  However,
     * alpha_real and alpha_imag will be always less than and usually comparable
     * with norm(A) in magnitude, and beta always less than and usually
     * comparable with norm(B).
     *
     * Real eigenvectors are stored in the columns of V, while imaginary
     * eigenvectors occupy adjacent columns of V with conjugate pairs
     * given by V(:,j) + i*V(:,j+1) and V(:,j) - i*V(:,j+1).
     *
     * @code
     *   template<typename PrecisionType, bool IsVectorBool>
     *   success_t GenEigenNonSymmetric(Matrix<PrecisionType, false> *A_garbage,
     *                                  Matrix<PrecisionType, false> *B_garbage,
     *                                  Matrix<PrecisionType, IsVectorBool> *alpha_real,
     *                                  Matrix<PrecisionType, IsVectorBool> *alpha_imag,
     *                                  Matrix<PrecisionType, IsVectorBool> *beta,
     *                                  Matrix<PrecisionType, IsVectorBool> *V_raw);
     *  // example
     *  fl:la::Matrix<double> a;
     *  fl::la::Random(4, 4, &a);
     *  fl::la::Matrix<double> b;
     *  fl::la::Random(4, 4, &b);
     *  fl::la::Matrix<double> alpha_real;
     *  alpha_real.Init(4, 1);
     *  fl::la::Matrix<double> alpha_imag;
     *  alpha_imag.Init(4, 1);
     *  fl::la::Matrix<double> beta;
     *  beta.Init(4, 1);
     *  fl::la::Matrix<double> v_raw;
     *  v_raw.Init(4, 1);
     *  fl::la::GenEigenNonsymmetric(&a, &b, &alpha_real,
     *         &alpha_image, &beta, &v_raw);
     * @endcode
     *
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is deduced by the function arguments
     * @param  IsVectorBool, boolean template parameter, for backward compatibility
     *                   with Vector. It is deduced by the funciton arguments
     * @param A_garbage an N-by-N matrix to be decomposed; overwritten
     *        with garbage
     * @param B_garbage an N-by-N matrix to be decomposed; overwritten
     *        with garbage
     * @param alpha_real a length-N array
     * @param alpha_imag a length-N array
     * @param beta a length-N array
     * @param V_raw an N-by-N matrix ptr to store eigenvectors, or NULL
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType, bool IsVectorBool>
    static void GenEigenNonSymmetric(Matrix<PrecisionType, false> *A_garbage,
                                     Matrix<PrecisionType, false> *B_garbage,
                                     Matrix<PrecisionType, IsVectorBool> *alpha_real,
                                     Matrix<PrecisionType, IsVectorBool> *alpha_imag,
                                     Matrix<PrecisionType, IsVectorBool> *beta,
                                     Matrix<PrecisionType, IsVectorBool> *V_raw,
                                     success_t *success) {
      DEBUG_MATSQUARE(*A_garbage);
      DEBUG_MATSQUARE(*B_garbage);
      DEBUG_ASSERT(A_garbage->n_rows() == B_garbage->n_rows());
      f77_integer info;
      f77_integer n = A_garbage->n_rows();
      const char *job = V_raw ? "V" : "N";
      PrecisionType d; // for querying optimal work size

      CppLapack<PrecisionType>::gegv("N", job, n, A_garbage->ptr(), n, B_garbage->ptr(), n,
                                     alpha_real->ptr(), alpha_imag->ptr(),
                                     beta->ptr(), NULL, 1, V_raw->ptr(), n, &d, -1, &info);
      {
        f77_integer lwork = (f77_integer)d;
        boost::scoped_array<PrecisionType> work(new PrecisionType[lwork]);

        CppLapack<PrecisionType>::gegv("N", job, n, A_garbage->ptr(), n, B_garbage->ptr(), n,
                                       alpha_real->ptr(), alpha_imag->ptr(),
                                       beta->ptr(), NULL, 1, V_raw->ptr(), n, work, lwork, &info);
      }

      *success =  SUCCESS_FROM_LAPACK(info);

    }

    /**
     * @brief Destructive SVD (A = U * S * VT).
     * Finding U and VT is optional (just pass NULL), but you must solve
     * either both or neither.
     *  template<typename PrecisionType>
     * @code
     *   template<typename PrecisionType>
     *   success_t SVDExpert(Matrix<PrecisionType, false>* A_garbage,
     *                       PrecisionType *s,
     *                       PrecisionType *U,
     *                       PrecisionType *VT);
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float double
     *
     * @param A_garbage an M-by-N matrix to be decomposed; overwritten
     *        with garbage
     * @param s a length-min(M,N) array to store the singluar values
     * @param U an M-by-min(M,N) matrix ptr to store left singular
     *        vectors, or NULL for neither U nor VT
     * @param VT a min(M,N)-by-N matrix ptr to store right singular
     *        vectors, or NULL for neither U nor VT
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static void SVDExpert(Matrix<PrecisionType, false>* A_garbage,
                          PrecisionType *s,
                          PrecisionType *U,
                          PrecisionType *VT,
                          success_t *success) {
      DEBUG_ASSERT_MSG((U == NULL) == (VT == NULL),
                       "You must fill both U and VT or neither.");
      f77_integer info;
      f77_integer m = A_garbage->n_rows();
      f77_integer n = A_garbage->n_cols();
      f77_integer k = std::min(m, n);
      boost::scoped_array<f77_integer> iwork(new f77_integer[8 * k]);
      const char *job = U ? "S" : "N";
      PrecisionType d; // for querying optimal work size

      CppLapack<PrecisionType>::gesdd(job, m, n, A_garbage->ptr(), m,
                                      s, U, m, VT, k, &d, -1, iwork.get(), &info);
      {
        f77_integer lwork = (f77_integer)d;
        // work for DGESDD can be large, we really do need to malloc it
        boost::scoped_array<PrecisionType> work(new PrecisionType[lwork]);

        CppLapack<PrecisionType>::gesdd(job, m, n, A_garbage->ptr(), m,
                                        s, U, m, VT, k, work.get(), lwork, iwork.get(), &info);
      }

      *success = SUCCESS_FROM_LAPACK(info);

    }

    /**
     * @brief Inits a vector and matrices to a singular value decomposition
     *        (A = U * S * VT).
     *
     * @code
     *   // Although SVD is a function, we use it as a class constructor, because
     *   // some of the template arguments have to be be explicitly defined
     *   // while others can be deduced. The following example will make it
     *   // more clear.
     *   template<MemoryAlloc M>
     *   class SVD {
     *    public:
     *     template<typename PrecisionType, bool IsVectorBool>
     *     SVD(const Matrix<PrecisionType, false> &A,
     *               Matrix<PrecisionType, IsVectorBool> *s);
     *     template<typename PrecisionType, bool IsVectorBool>
     *     SVD(const Matrix<PrecisionType, false> &A,
     *               Matrix<PrecisionType, IsVectorBool> *s,
     *               Matrix<PrecisionType, false> *U,
     *               Matrix<PrecisionType, false> *VT);
     *     success_t success;
     *   };
     *   // example
     *   fl::la::Matrix<double> a;
     *   // assign A to [0 1; -1 0]
     *   a.Init(2, 2);
     *   a[0]=0; a[1]=-1; a[2]=1; a[3]=0;
     *   fl::la::Matrix<double> s; // Not initialized
     *   fl::la::Matrix<double> U, VT; // Not initialized
     *   success_t success = fl::la::SVD<fl::la::Init>(a, &s, &U, &VT).success;
     *   // s is now [1 1]
     *   // U is now [0 1; 1 0]
     *   //  V is now [-1 0; 0 1]
     *   fl::la::Matrix<double> S;
     *   S.Init(s.length(), s.length());
     *   S.SetDiagonal(s);
     *   fl::la::Matrix<double> tmp, result;
     *   fl::la::Mul<fl::la::Init>(U, S, &tmp);
     *   fl::la::MulInit(tmp, VT, &result);
     *   // A and result should be equal (but for round-off)
     * @endcode
     * @param M, if it is fl::la::Init, then the function initializes the results
     *           if it is fl::la::Overwrite, then the function just overwrites
     *           the already allocated results
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced by the function arguments
     * @param IsVectorBool, bool template argument for backward compatibility with
     *                  Vector. It is deduced by the function arguments
     * @param A an M-by-N matrix to be decomposed
     * @param s a fresh vector to be initialized to length min(M,N)
     *        and filled with the singular values
     * @param U a fresh matrix to be initialized to size M-by-min(M,N)
     *        and filled with the left singular vectors
     * @param VT a fresh matrix to be initialized to size min(M,N)-by-N
     *        and filled with the right singular vectors
     * @return SUCCESS_PASS if successful, SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M,
             fl::la::TransMode IsTransA = fl::la::NoTrans>
    class SVD {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        SVD(const Matrix<PrecisionType, false> &A,
            Matrix<PrecisionType, IsVectorBool> *s,
            success_t *success) {
          AllocationTrait<M>::Init(std::min(A.n_rows(), A.n_cols()), s);
          Matrix<PrecisionType, false> tmp;
          tmp.Copy(A);
          SVDExpert<PrecisionType>(&tmp, s->ptr(), NULL, NULL, success);
        }

        template<typename PrecisionType, bool IsVectorBool>
        SVD(const Matrix<PrecisionType, false> &A,
            Matrix<PrecisionType, IsVectorBool> *s,
            Matrix<PrecisionType, false> *U,
            Matrix<PrecisionType, false> *VT,
            success_t *success) {
          index_t k = std::min(A.n_rows(), A.n_cols());
          AllocationTrait<M>::Init(k, s);
          AllocationTrait<M>::Init(A.n_rows(), k, U);
          AllocationTrait<M>::Init(k, A.n_cols(), VT);
          Matrix<PrecisionType, false> tmp;
          tmp.Copy(A);
          SVDExpert(&tmp, s->ptr(), U->ptr(), VT->ptr(), success);
        }
    };

    /**
     * @brief Destructively computes the Cholesky factorization (A = U' * U).
     * @code
     *   template<typename PrecisionType>
     *   success_t CholeskyExpert(Matrix<PrecisionType, false> *A_in_U_out);
     *   // example
     *   fl::la::Matrix<double> a_in_u_out;
     *   a.Init(2, 2);
     *   a[0]=4; a[1]=1; a[2]=1; a[3]=2;
     *   fl::la::CholeskyExpert(&a_in_u_out);
     * @endcode
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced by the function arguments
     * @param A_in_U_out an N-by-N matrix to factorize; overwritten
     *        with result
     * @return SUCCESS_PASS if the matrix is symmetric positive definite,
     *         SUCCESS_FAIL otherwise
     */
    template<typename PrecisionType>
    static void CholeskyExpert(Matrix<PrecisionType, false> *A_in_U_out, success_t *success) {
      DEBUG_MATSQUARE(*A_in_U_out);
      f77_integer info;
      f77_integer n = A_in_U_out->n_rows();

      CppLapack<PrecisionType>::potrf("U", n, A_in_U_out->ptr(), n, &info);

      /* set the garbage part of the matrix to 0. */
      for (f77_integer j = 0; j < n; j++) {
        ::memset(A_in_U_out->GetColumnPtr(j) + j + 1, 0, (n - j - 1) * sizeof(PrecisionType));
      }

      *success = SUCCESS_FROM_LAPACK(info);
    }


    /**
     * @brief Inits a matrix to the Cholesky factorization (A = U' * U).
     *
     * @code
     *   // Although Cholesky is a function, we use it here as a class constructor
     *   // because some of the template arguments have to be explicitly defined
     *   // while others can be deduced. The following example will make it more clear
     *
     *   template<MemoryAlloc M>
     *   class Cholesky {
     *    public:
     *     template<typename PrecisionType>
     *     Cholesky(const Matrix<PrecisionType, false> &A,
     *                    Matrix<PrecisionType, false> *U);
     *     success_t success;
     *   };
     *   // example
     *   fl::la::Matrix<double> a;
     *   //  assign a to [1 1; 1 2]
     *   a.Init(2, 2);
     *   a[0]=1; a[1]=1; a[2]=1; a[3]=2;
     *   fl::la::Matrix<double> u; // Not initialized
     *   success_t success = fl::la::Cholesky<fl::la::Init>(a, &u).success;
     *   // u is now [1 1; 0 1]
     *   fl::la::Matrix<double> result;
     *   fl::la::Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(U, U, result);
     *   // A and result should be equal (but for round-off)
     * @endcode
     * @param M, if it is fl::la::Init, then the function initializes the results
     *           if it is fl::la::Overwrite, then the function just overwrites
     *           the already allocated results
     * @param PrecisionType, template parameter for the precision, currently supports
     *        float, double. It is automatically deduced by the function arguments
     *
     * @param A an N-by-N matrix to factorize
     * @param U a fresh matrix to be initialized to size N-by-N
     *        and filled with the factorization
     * @return SUCCESS_PASS if the matrix is symmetric positive definite,
     *         SUCCESS_FAIL otherwise
     */
    template<fl::la::MemoryAlloc M>
    class Cholesky {
      public:
        template<typename PrecisionType>
        Cholesky(const Matrix<PrecisionType, false> &A,
                 Matrix<PrecisionType, false> *U, success_t *success) {
          AllocationTrait<M>::Init(A.n_rows(), A.n_cols(), U);
          U->CopyValues(A);
          CholeskyExpert(U, success);
        }
    };

    /*
     * TODO:
     *   symmetric eigenvalue
     *   http://www.netlib.org/lapack/lug/node71.html
     *   dgels for least-squares problems
     *   dgesv for linear equations
     */

    /**
     * Finds the Euclidean distance squared between two vectors.
     */
    template<typename PrecisionType>
    static inline PrecisionType DistanceSqEuclidean(
      index_t length, const PrecisionType *va, const PrecisionType *vb) {
      PrecisionType s = 0;
      do {
        PrecisionType d = (*va) - (*vb);
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
    template<typename PrecisionType, bool IsVectorBool>
    static inline PrecisionType DistanceSqEuclidean(const Matrix<PrecisionType, IsVectorBool>& x,
        const Matrix<PrecisionType, IsVectorBool>& y) {
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
    template<int t_pow>
    class RawLMetric {
      public:
        template < typename PointType1, typename PointType2,
        typename CalcPrecisionType >
        RawLMetric(const PointType1 &va, const PointType2 &vb,
                   CalcPrecisionType *result) {
          DEBUG_ASSERT(va.size() == vb.size());
          size_t length = va.size();
          typedef CalcPrecisionType CalcPrecision_t;
          *result = 0;
          index_t i = 0;
          do {
            CalcPrecision_t d = va[i] - vb[i];
            i++;
            *result += math::PowAbs<CalcPrecision_t, t_pow, 1>(d);
          }
          while (--length);
        }

        template < typename PointType1, typename PointType2,
        typename PointType3, typename CalcPrecisionType >
        RawLMetric(const PointType3 &w,
                   const PointType1 &va, const PointType2 &vb,
                   CalcPrecisionType *result) {
          DEBUG_ASSERT(va.size() == vb.size());
          DEBUG_ASSERT(w.size() == va.size());
          size_t length = va.size();
          typedef CalcPrecisionType CalcPrecision_t;
          *result = 0;
          index_t i = 0;
          do {
            CalcPrecision_t d = va[i] - vb[i];
            *result += w[i] * math::PowAbs<CalcPrecision_t, t_pow, 1>(d);
            i++;
          }
          while (--length);
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
    template<typename PrecisionType, int t_pow>
    static inline PrecisionType LMetric(
      index_t length, const PrecisionType *va, const PrecisionType *vb) {
      PrecisionType result;
      RawLMetric<t_pow>(length, va, vb, &result);
      return math::Pow<PrecisionType, 1, t_pow>(result);
    }
    /** Finds the trace of the matrix.
     *  Trace(A) is the sum of the diagonal elements
     */
    template<typename PrecisionType, bool IsVectorBool>
    static inline PrecisionType Trace(Matrix<PrecisionType, IsVectorBool> &a) {
      // trace has meaning only for square matrices
      DEBUG_SAME_SIZE(a.n_cols(), a.n_rows());
      PrecisionType trace = 0;
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
    template<fl::la::MemoryAlloc M>
    class LeastSquareFit {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        LeastSquareFit(const Matrix<PrecisionType, IsVectorBool> &y,
                       const Matrix<PrecisionType, false> &x,
                       Matrix<PrecisionType, IsVectorBool> *a,
                       success_t *success) {
          DEBUG_SAME_SIZE(y.n_rows(), x.n_rows());
          DEBUG_ASSERT(x.n_rows() >= x.n_cols());
          Matrix<PrecisionType, true> r_xy_mat;
          Matrix<PrecisionType, false> r_xx_mat;
          Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(x, x, &r_xx_mat);
          Mul<fl::la::Init, fl::la::Trans, fl::la::NoTrans>(x, y, &r_xy_mat);
          Solve<M>(r_xx_mat, r_xy_mat, a, success);
          if unlikely(*success != SUCCESS_PASS) {
            if (*success == SUCCESS_FAIL) {
              fl::logger->Warning()<<"Least square fit failed ";
            }
            else {
              fl::logger->Warning()<< "Least square fit returned a warning ";
            }
          }
          return ;
        }
    };
    /** Solves the classic least square problem y=x'*a
     *  where y  is N x r
     *        x  is m x N
     *        a  is m x r
     *  We require that N >= m
     *  a should not be initialized
     */
    template<fl::la::MemoryAlloc M>
    class LeastSquareFitTrans {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        LeastSquareFitTrans(Matrix<PrecisionType, IsVectorBool> &y,
                            Matrix<PrecisionType, false> &x,
                            Matrix<PrecisionType, IsVectorBool> *a,
                            success_t *success) {
          DEBUG_SAME_SIZE(y.n_rows(), x.n_cols());
          DEBUG_ASSERT(x.n_cols() >= x.n_rows());
          Matrix<PrecisionType, false> r_xy_mat;
          Matrix<PrecisionType, false> r_xx_mat;
          Mul<fl::la::Init, fl::la::NoTrans, fl::la::Trans>(x, x, &r_xx_mat);
          Mul<fl::la::Init, fl::la::NoTrans, fl::la::NoTrans>(x, y, &r_xy_mat);
          Solve<M>(r_xx_mat, r_xy_mat, a, success);
          if unlikely(*success != SUCCESS_PASS) {
            if (*success == SUCCESS_FAIL) {
              fl::logger->Die()<<"Least square fit failed \n";
            }
            else {
              fl::logger->Warning()<<"Least square fit returned a warning ";
            }
          }
          return ;
        }
    };
    /**
     * @brief Entrywise element multiplication of two memory blocks
     * Also known as the Hadamard product
     * In Matlab this is the .* operator
     * (\f$ C \gets A \verb@.*@ B\f$).
     *
     * @param length The length of each memory block.
     * @param A The pointer to the starting memory block of the first.
     * @param B The pointer to the starting memory block of the second.
     * @param C The resulting Hadamard product.
     */
    template<typename PrecisionType >
    static void DotMulExpert(const index_t length,
                             const PrecisionType *A,
                             const PrecisionType *B,
                             PrecisionType *C) {
      CppBlas<PrecisionType>::gbmv("N",
                                   length, length, 0, 0, 1, A, 1, B, 1, 0, C, 1);
    }

    /**
     * @brief Entrywise element multiplication of two matrices
     * Also known as the Hadamard product
     * In Matlab this is the .* operator
     * (\f$ C \gets A \verb@.*@ B\f$).
     * @code
     *   // example
     *   fl::la::Matrix<double> a;
     *   fl::la::Matrix<double> b;
     *   fl::la::Random(3, 1, &a);
     *   fl::la::Random(3, 1, &b);
     *   fl::la::Matrix<double> c;
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

    template<fl::la::MemoryAlloc MemAlloc>
    class DotMul {
      public:
        template<typename PrecisionType, bool IsVectorBool>
        DotMul(const Matrix<PrecisionType, IsVectorBool> &A,
               const Matrix<PrecisionType, IsVectorBool> &B,
               Matrix<PrecisionType, IsVectorBool> *C) {
          DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
          DEBUG_SAME_SIZE(A.n_cols(), B.n_cols());
          AllocationTrait<MemAlloc>::Init(A.n_rows(), A.n_cols(), C);
          DotMulExpert(A.n_elements(), A.ptr(), B.ptr(), C->ptr());
        }
    };


    /**
     * @brief Entrywise element multiplication of two memory blocks
     * Also known as the Hadamard product
     * In Matlab this is the .* operator
     * (\f$ A \gets A \verb@.*@ B\f$).
     *
     * @param length The length of each memory block.
     * @param A The left hand side but will be overwritten by A .* B
     * @param B The right hand side.
     */
    template<typename PrecisionType>
    static void DotMulTo(const index_t length,
                         const PrecisionType *B,
                         PrecisionType *A) {

      Matrix<PrecisionType, true> C;
      C.Init(length);
      DotMulExpert(length, A, B, C.ptr());
      memcpy(A, C.ptr(), length * sizeof(PrecisionType));
    }
    /**
     * @brief Entrywise element multiplication of two matrices
     * Also known as the Hadamard product
     * In Matlab this is the .* operator
     * (\f$ A \gets A \verb@.*@ B\f$).
      * @code
     *   // example
     *   fl::la::Matrix<float> a;
     *   fl::la::Matrix<float> b;
     *   fl::la::Random(3, 4, &a);
     *   fl::la::Random(3, 4, &b;
     *   fl::la::DotMulTo(&a, b);
     * @endcode
     * @param A, matrix
     * @param B, matrix
     */

    template<typename PrecisionType, bool IsVectorBool>
    static void DotMulTo(const Matrix<PrecisionType, IsVectorBool> &B,
                         Matrix<PrecisionType, IsVectorBool> *A) {

      DEBUG_SAME_SIZE(A->n_rows(), B.n_rows());
      DEBUG_SAME_SIZE(A->n_cols(), B.n_cols());
      DotMulTo(A->n_elements(), B.ptr(), A->ptr());
    }

    /**
     * @brief Elementwise integer powers of memory blocks
     * (\f$ B \gets A\verb@.^@n \f$)
    */
    template<typename PrecisionType>
    static void DotIntPowExpert(const index_t length,
                                index_t power,
                                const PrecisionType *A,
                                PrecisionType *B) {
      memcpy(B, A, length * sizeof(PrecisionType));

      // Note: this is very important! You loop only (n - 1) times
      // because B contains a copy of A.
      for (index_t i = 1; i < power; i++) {
        DotMulTo(length, A, B);
      }
    }

    /**
     * @brief Elementwise integer powers of matrices
     * (\f$ B \gets A\verb@.^@n \f$)
      * @code
     *   fl::la::Matrix<float> a;
     *   fl::la::Random(5, 6, &a);
     *   fl::la::Matrix<float> b;
     *   fl::la::DotIntPow<fl::la::Init>(4, a, &b);
     * @endcode
     *
     * @param power, the exponent
     * @param A,  matrix
     * @param B, matrix, the result
     */
    template<fl::la::MemoryAlloc MemAlloc>
    class DotIntPow {
      public:
        template<typename PrecisionType>
        DotIntPow(index_t power,
                  const Matrix<PrecisionType, false> &A,
                  Matrix<PrecisionType, false> *B) {
          AllocationTrait<MemAlloc>::Init(A.n_rows(), A.n_cols(), B);
          DotIntPowExpert(A.n_elements(), power, A.ptr(), B->ptr());
        }
    };



    /**
     * @brief Elementwise integer powers of memory blocks
     * (\f$ A \gets A\verb@.^@n \f$)
     */
    template<typename PrecisionType>
    static void DotIntPowTo(const index_t length,
                            index_t power,
                            PrecisionType *A) {
      Matrix<PrecisionType, true> temp;
      temp.Init(length);
      DotIntPowExpert(length, power, A, temp.ptr());
      memcpy(A, temp.ptr(), length * sizeof(PrecisionType));
    }

    /**
     * @brief Elementwise integer powers of matrices
     * (\f$ A \gets A\verb@.^@n \f$)
     * @code
     *  fl::la::Matrix<double> a;
     *  fl::la::Random(3, 5, &a);
     *  fl::la::DoIntPowTo(3, &a);
     * @endcode
     * @param a, matrix
     * @param power, the exponent
     */
    template<typename PrecisionType, bool IsVectorBool>
    static void DotIntPowTo(index_t power,
                            Matrix<PrecisionType, IsVectorBool> *A) {
      DotIntPowTo(A->n_elements(), power, A->ptr());
    }

    /**
      * @brief In this section we provide some basic object functions for
      * matrix processing
      */
    template<typename PrecisionType>
    class Sin {
      public:
        Sin() {
          frequency_ = 1.0;
        }
        void Init(PrecisionType frequency) {
          frequency_ = frequency;
        }
        void set(PrecisionType frequency) {
          frequency_ = frequency;
        }
        PrecisionType operator()(PrecisionType x) {
          return sin(frequency_ * x);
        }
      private:
        PrecisionType frequency_;
    };

    template<typename PrecisionType>
    class Cos {
      public:
        Cos() {
          frequency_ = 1.0;
        }
        void Init(PrecisionType frequency) {
          frequency_ = frequency;
        }
        void set(PrecisionType frequency) {
          frequency_ = frequency;
        }
        PrecisionType operator()(PrecisionType x) {
          return cos(frequency_ * x);
        }
      private:
        PrecisionType frequency_;
    };

    template<typename PrecisionType>
    class Tan {
      public:
        Tan() {
          frequency_ = 1.0;
        }
        void Init(PrecisionType frequency) {
          frequency_ = frequency;
        }
        void set(PrecisionType frequency) {
          frequency_ = frequency;
        }
        PrecisionType operator()(PrecisionType x) {
          return tan(frequency_ * x);
        }
      private:
        PrecisionType frequency_;
    };

    template<typename PrecisionType>
    class Exp {
      public:
        Exp() {
          alpha_ = 1.0;
        }
        void Init(PrecisionType alpha) {
          alpha_ = alpha;
        }
        void set(PrecisionType alpha) {
          alpha_ = alpha;
        }
        PrecisionType operator()(PrecisionType x) {
          return exp(alpha_ * x);
        }
      private:
        PrecisionType alpha_;
    };

    template<typename PrecisionType>
    class Log {
      public:
        Log() {
          alpha_ = 1.0;
        }
        void Init(PrecisionType alpha) {
          alpha_ = alpha;
        }
        void set(PrecisionType alpha) {
          alpha_ = alpha;
        }
        PrecisionType operator()(PrecisionType x) {
          DEBUG_ASSERT(x>0);
          return log(alpha_ * x);
        }
      private:
        PrecisionType alpha_;
    };

    template<typename PrecisionType>
    class Pow {
      public:
        Pow() {
          alpha_ = 1.0;
        }
        void Init(PrecisionType alpha) {
          alpha_ = alpha;
        }
        void set(PrecisionType alpha) {
          alpha_ = alpha;
        }
        PrecisionType operator()(PrecisionType x) {
          return pow(alpha_, x);
        }
      private:
        PrecisionType alpha_;
    };

    /**
    * @brief Elementwise fun of a memory block
    *        (\f$ B = {\tt fun}(A) \f$)
    *        where fun is an arbitrary function that operates on
    *        a single element.
    *        fun is passed as a function object. It is a class that overloads the
    *        operator()
    */
    template<typename PrecisionType, typename Function>
    static void DotFunExpert(const index_t length,
                             Function &fun,
                             const PrecisionType *A,
                             PrecisionType *B) {
      for (index_t i = 0; i < length; i++) {
        B[i] = fun(A[i]);
      }
    }
    /**
     * @brief Elementwise fun of a matrix
     *        (\f$ B= {\tt fun}(A) \f$)
     *        where fun is an arbitrary function that operates on
     *        a single element.
     *        fun is passed as a function object. It is a class that overloads the
     *        operator()
     * @brief Elementwise fun of two matrices
     *        (\f$ C= {\tt fun}(A, B) \f$)
     *        where fun is an arbitrary function that operates on
     *        a single element.
     *        fun is passed as a function object. It is a class that overloads the
     *        operator()
     *
     * @code
     *   // example
     *   fl:;la::Matrix<double> a;
     *   fl::la::Random(6, 7, &a);
     *   fl::la::Cos fun;
     *   fun.Init(3.14);
     *   fl::la::Matrix<double> b;
     *   // b= cos(a);
     *   fl::la::DotFun<fl::la::Init>(fun, a, &b);
     *   // c= fun1(a, b);
     *   fl::la::Matrix<double> c;
     *   // MyFun takes as inputs two matrices
     *   Myfun  fun1;
     *   fl::la::DotFun(fun, a, b, &c);
     * @endcode
     */
    template<fl::la::MemoryAlloc MemAlloc>
    class DotFun {
      public:
        template<typename Function, typename PrecisionType, bool IsVectorBool>
        DotFun(Function &fun,
               const Matrix<PrecisionType, IsVectorBool> &A,
               Matrix<PrecisionType, IsVectorBool> *B) {
          AllocationTrait<MemAlloc>::Init(A.n_rows(), A.n_cols(), B);
          DotFunExpert<PrecisionType, Function>(A.n_elements(), fun, A.ptr(), B->ptr());
        }
        template<typename Function, typename PrecisionType, bool IsVectorBool>
        DotFun(Function &fun,
               const Matrix<PrecisionType, IsVectorBool> &A,
               const Matrix<PrecisionType, IsVectorBool> &B,
               const Matrix<PrecisionType, IsVectorBool> *C) {
          DEBUG_SAME_SIZE(A.n_rows(), B.n_rows());
          DEBUG_SAME_SIZE(A.n_cols(), B.n_cols());
          AllocationTrait<MemAlloc>::Init(A.n_rows(), A.n_cols(), C);
          for (index_t i = 0; i < A.n_elements(); i++) {
            C[i] = fun(A[i], B[i]);
          }
        }
    };

    /**
     * @brief Sums the elements of a memory block
     *   (\f$ {\tt sum}(A)\f$)
     *   Notice: Because summation can lead to overflow we have a second
     *   template parameter called RetPrecisionType for defining the Precision
     *   of the sum.
     */
    template<typename PrecisionType, typename RetPrecisionType>
    static void SumExpert(const index_t length,
                          const PrecisionType *A, RetPrecisionType *sum) {
      *sum = 0;
      for (index_t i = 0; i < length; i++) {
        (*sum) += A[i];
      }
    }

    /**
     * @brief it sums all the elements of matrix A
     * (\f$ {\tt sum}({\tt sum}(A))\f$)
     * @code
     *   // example
     *   fl::la::Matrix<float> a;
     *   fl::la::Random(100, 1000, &a);
     *   double mysum;
     *   Sum(a, &mysum);
     * @endcode
     */
    class Sum {
      public:
        template<typename PrecisionType, typename RetPrecisionType, bool IsVectorBool>
        Sum(const Matrix<PrecisionType, IsVectorBool> &A, RetPrecisionType *sum) {
          SumExpert(A.n_elements(), A.ptr(), sum);
        }
    };

    /**
     * @brief it sums all the columns of matrix A
     *        (\f$ {\tt sum}(A)\f$)
     * @code
     *   fl::la::Matrix<float> a;
     *   fl::la::Random(40, 70, &a);
     *   fl::la::Matrix< col_sums;
     *   fl::la::SumCols<fl::la::Init>(a, &col_sums);
     * @endcode
     */
    template<fl::la::MemoryAlloc MemAlloc>
    class SumCols {
      public:
        template<typename PrecisionType, typename RetPrecisionType, bool IsVector>
        SumCols(const Matrix<PrecisionType, false> &A,
                Matrix<RetPrecisionType, IsVector> *col_sums) {
          AllocationTrait<MemAlloc>::Init(A.n_cols(), col_sums);
          RetPrecisionType *ptr = col_sums->ptr();
          for (index_t i = 0; i < A.n_cols(); i++) {
            ptr[i] = SumExpert(A.n_rows(),
                               A.GetColumnPtr(i));
          }
        }
    };

    /**
     * @brief it sums all the rows of matrix A
     *        (\f$ {\tt sum}(A,2)\f$)
     * @code
     *   fl::la::Matrix<double> a;
     *   fl::la::Random(50, 60, &a);
     *   fl::la::Matrix<long double> row_sums;
     *   fl::la::SumRows<fl::la::Init>(a, &row_sums);
     * @endcode
     */
    template<fl::la::MemoryAlloc MemAlloc>
    class SumRows {
      public:
        template<typename PrecisionType, typename RetPrecisionType, bool IsVectorBool >
        SumRows(const Matrix<PrecisionType, false> &A,
                Matrix<RetPrecisionType, IsVectorBool> *row_sums) {
          AllocationTrait<MemAlloc>::Init(A.n_rows(), row_sums);
          RetPrecisionType *ptr = row_sums->ptr();
          for (index_t i = 0; i < A.n_rows(); i++) {
            ptr[i] = 0;
            for (index_t j = 0; j < A.n_cols(); j++) {
              ptr[i] += A.get(i, j);
            }
          }
        }
    };

    /**
     * @brief Multiplies the elements of a memory block
     *        (\f$ {\tt prod}(A) \f$)
     *        Notice: Because product can lead to overflow we have a second
     *        template parameter called RetPrecisionType for defining the Precision
     *        of the sum.
     */
    template<typename PrecisionType, typename RetPrecisionType>
    static void ProdExpert(const index_t length,
                           const PrecisionType *A,
                           RetPrecisionType *prod) {
      *prod = 1;
      for (index_t i = 0; i < length; i++) {
        (*prod) *= A[i];
      }
    }

    /**
     * @brief it  multiplies all the elements of matrix A
     *        (\f$ {\tt prod}({\tt prod}(A))\f$)
     * @code
     *   fl::la::GenMetrix<double> a;
     *   fl::la::Random(500, 1000, &a);
     *   long double prod;
     *   fl::la::Prod(a, &prod);
     *
     * @endcode
     */
    class Prod {
      public:
        template<typename PrecisionType, typename RetPrecisionType, bool IsVectorBool>
        Prod(const Matrix<PrecisionType, IsVectorBool> &A, RetPrecisionType *prod) {
          *prod = ProdExpert<PrecisionType, RetPrecisionType>(A.n_elements(), A.ptr());
        }
    };

    /**
     * @brief it prods all the columns of matrix A
     *        (\f$ {\tt prod}(A)\f$)
     * @code
     *  fl::la::Matrix<float> a;
     *  fl::la::Random(40, 100, &a);
     *  fl::la::Matrix<double> prod_cols;
     *  fl::la::ProdCols<fl::la::Init>(a, &prod_cols);
     * @endcode
     */
    template<fl::la::MemoryAlloc MemAlloc>
    class ProdCols {
      public:
        template<typename PrecisionType, typename RetPrecisionType, bool IsVectorBool>
        ProdCols(const Matrix<PrecisionType, false> &A,
                 Matrix<RetPrecisionType, IsVectorBool> *col_prods) {
          AllocationTrait<MemAlloc>::Init(A.n_cols(), col_prods);
          RetPrecisionType *ptr = col_prods->ptr();
          for (index_t i = 0; i < A.n_cols(); i++) {
            ptr[i] = ProdExpert(A.n_rows(),
                                A.GetColumnPtr(i));
          }
        }
    };

    /**
     * @brief it multiplies all the rows of matrix A
     *        (\f$ {\tt prod}(A,2)\f$)
     * @code
     *   fl::la::Matrix<float> a;
     *   fl::la::Random(49, 299, &a);
     *   fl::la::Matrix<long double> prod_rows;
     *   fl::la::ProdRows(a, &prod_rows);
     * @endcode
     */
    template<fl::la::MemoryAlloc MemAlloc>
    class ProdRows {
      public:
        template<typename PrecisionType, typename RetPrecisionType, bool IsVectorBool>
        ProdRows(const Matrix<PrecisionType, false> &A,
                 Matrix<RetPrecisionType, IsVectorBool> *row_prods) {
          AllocationTrait<MemAlloc>::Init(A.n_rows(), row_prods);
          RetPrecisionType *ptr = row_prods->ptr();
          for (index_t i = 0; i < A.n_prods(); i++) {
            ptr[i] = 1;
            for (index_t j = 0; j < A.n_cols(); j++) {
              ptr[i] *= A.get(i, j);
            }
          }
        }
    };



    /**
     * @brief it sums all the elements of matrix A preprocessed with a function
     *        (\f$ {\tt sum}({\tt sum}({\tt fun}(A)))\f$)
     *        Class Function is a function object so it must implement
     *        the operator()
     * @code
     *  fl::la::Matrix<double> a;
     *  fl::la::Random(40, 200, &a);
     *  fl::la::Tan fun;
     *  fun.Init(3.13);
     *  double fun_sum;
     *  fl::la::FunSum(fun, a, &fun_sum);
     * @endcode
     */
    template < typename PrecisionType, typename RetPrecisionType,
    bool IsVectorBool,
    typename Function >
    static void FunSum(Function &fun,
                       const Matrix<PrecisionType, false> &A,
                       RetPrecisionType *sum) {
      *sum = 0;
      for (index_t i = 0; i < A.n_elements(); i++) {
        (*sum) = fun(A[i]);
      }
    }

    /**
     * @brief it sums all the columns of matrix A preprocessed with a function
     *        (\f$ {\tt sum}({\tt fun}(A))\f$)
     *        Class Function is a function object so it must implement
     *        the operator()
     * @code
     *   fl::la::Matrix<double> a;
     *   fl::la::Random(40, 40, &a);
     *   fl::la::Sin fun;
     *   fun.Init(6.28);
     *   fl::la::Matrix<double> fun_sum_cols;
     *   fl::la::FunSumCols(fun, a, &fun_sum_cols);
     * @endcode
     */
    template<fl::la::MemoryAlloc MemAlloc>
    class FunSumCols {
      public:
        template < typename PrecisionType, typename RetPrecisionType,
        bool IsVectorBool, typename Function >
        FunSumCols(Function &fun,
                   const Matrix<PrecisionType, false> &A,
                   Matrix<RetPrecisionType, IsVectorBool> *col_sums) {
          AllocationTrait<MemAlloc>::Init(A.n_cols(), col_sums);
          RetPrecisionType *ptr = col_sums->ptr();
          for (index_t i = 0; i < A.n_cols(); i++) {
            ptr[i] = FunSum<PrecisionType, RetPrecisionType, MemAlloc, Function>(A.n_rows(),
                     fun, A.GetColumnPtr(i));
          }
        }
    };

    /**
     * @brief it sums all the rows of matrix A preprocessed with a function
     *        (\f$ {\tt sum}({\tt fun}(A))\f$)
     *        Class Function is a function object so it must implement
     *        the operator()
     * @code
     *   fl::la::Matrix<double> a;
     *   fl::la::Random(40, 40, &a);
     *   fl::la::Sin fun;
     *   fun.Init(6.28);
     *   fl::la::Matrix<double> fun_sum_cols;
     *   fl::la::FunSumRows(fun, a, &fun_sum_cols);
     * @endcode
     */
    template<fl::la::MemoryAlloc MemAlloc>
    class FunSumRows {
      public:
        template < typename PrecisionType, typename RetPrecisionType,
        bool IsVectorBool, typename Function >
        FunSumRows(Function &fun,
                   const Matrix<PrecisionType, false> &A,
                   Matrix<RetPrecisionType, IsVectorBool> *row_sums) {
          AllocationTrait<MemAlloc>::Init(A.n_rows(), row_sums);
          RetPrecisionType *ptr = row_sums->ptr();
          for (index_t i = 0; i < A.n_rows(); i++) {
            ptr[i] = 0;
            for (index_t j = 0; j < A.n_cols(); j++) {
              ptr[i] += fun(A.get(i, j));
            }
          }
        }
    };

    /**
     *  @brief Matlab like utility function for returning a matrix initialized
     *  with zeros
     *
     */
    template<typename PrecisionType>
    static void Zeros(const index_t rows,
                      const index_t cols,
                      Matrix<PrecisionType, false> *A) {
      A->Init(rows, cols, A);
      A->SetAll(PrecisionType(0.0));
    }

    /**
     *  @brief Matlab like utility function for returning a vector initialized
     *  with zeros
     *
     */
    template<typename PrecisionType>
    static void Zeros(const index_t length,
                      Matrix<PrecisionType, true> *A) {
      A->Init(length);
      A->SetAll(0.0);
    }

    /**
     *  @brief Matlab like utility function for returning a matrix initialized
     *  with ones
     *
     */
    template<typename PrecisionType>
    static void Ones(const index_t rows,
                     const index_t cols,
                     Matrix<PrecisionType, false> *A) {
      A->Init(rows, cols, A);
      A->SetAll(PrecisionType(1.0));
    }

    /**
     *  @brief Matlab like utility function for returning a vector initialized
     *  with ones
     *
     */
    template<typename PrecisionType>
    static void Ones(const index_t length,
                     Matrix<PrecisionType, true> *A) {
      A->Init(length);
      A->SetAll(1.0);
    }

    /**
     *  @brief Matlab like utility function for returning a matrix initialized
     *  with random numbers in [lo, hi] interval
     *
     */
    template<typename PrecisionType>
    static void Rand(const index_t rows,
                     const index_t cols,
                     const index_t lo,
                     const index_t hi,
                     Matrix<PrecisionType, false> *A) {
      A->Init(rows, cols);
      for (index_t i = 0; i < rows; i++) {
        for (index_t j = 0; j < cols; j++) {
          A->set(i, j, math::Random(lo, hi));
        }
      }
    }
    /**
     *  @brief Matlab like utility function for returning a symmetric matrix initialized
     *  with random numbers in [lo, hi] interval
     *
     */
    template<typename PrecisionType>
    static void RandSymmetric(const index_t rows,
                              const index_t lo,
                              const index_t hi,
                              Matrix<PrecisionType, false> *A) {
      A->Init(rows, rows);
      for (index_t i = 0; i < rows; i++) {
        for (index_t j = 0; j <= i; j++) {
          PrecisionType value = math::Random(lo, hi);
          A->set(i, j, value);
          A->set(j, i, value);
        }
      }
    }
    /**
      *  @brief Matlab like utility function for returning a matrix initialized
      *  with random numbers in [0, 1] interval
      *
      */
    template<typename PrecisionType>
    static void Random(const index_t rows,
                       const index_t cols,
                       Matrix<PrecisionType, false> *A) {
      Rand<PrecisionType>(rows, cols, 0, 1, A);
    }

    /**
      *  @brief Matlab like utility function for returning a symmetric matrix initialized
      *  with random numbers in [0, 1] interval
      *
      */
    template<typename PrecisionType>
    static void RandomSymmetric(const index_t rows,
                                Matrix<PrecisionType, false> *A) {
      RandSymmetric<PrecisionType>(rows, 0, 1, A);
    }

    /**
     *  @brief Matlab like utility function for generating  identity matrix
     *
     */
    template<typename PrecisionType>
    static void Eye(index_t dimension,
                    Matrix<PrecisionType, false> *I) {
      I->InitDiagonal(dimension, PrecisionType(1.0));
    }

    /**
     * @brief Computes the mean value of a vector.
     *        if it is a matrix it computes the mean for every column
     *        just like matlab
     * @code
     *  fl::la::Matrix<double> a;
     *  fl::la::Matrix<double> means;
     *  fl::la::Random(30, 23, &a);
     *  fl::la::Mean<fl::la::Init>(a, &means);
     *
     * @endcode
     * @param A the input vector
     * @return the mean value
     */
    template<fl::la::MemoryAlloc M>
    class Mean {
      public:
        template<typename PrecisionType, typename Container, bool IsVectorBool>
        Mean(Matrix<PrecisionType, IsVectorBool> &A, Container *mean) {
          You_have_a_precision_conflict <
          typename fl::TypeInfo<Container>::Precision_t, PrecisionType > ();

          AllocationTrait<M>::Init(A.n_cols(), mean);
          if (IsVectorBool == true || A.n_cols() == 1) {
            PrecisionType c = 0.0;
            index_t n = A.length();
            for (index_t i = 0; i < n; i++) {
              c = c + A[i];
            }
            mean[0] = c / n;
          }
          else {
            index_t n = A.n_rows();
            for (index_t i = 0; i < A.n_cols(); i++) {
              PrecisionType c = 0.0;
              for (index_t j = 0; j < n; j++) {
                c = c + A.get[j, i];
              }
              mean[i] = c;
            }
          }
        }
    };

    /**
     * @brief Computes the variance of a vector using "corrected two-pass algorithm".
     *        See "Numerical Recipes in C" for reference.
     * @code
     *  fl::la::Matrix<double> a;
     *  fl::la::Matrix<double> vars;
     *  fl::la::Random(30, 23, &a);
     *  fl::la::Var<fl::la::Init>(a, &var);
     *
     * @endcode
     * @param A the input vector
     * @return the variance
     */
    template<fl::la::MemoryAlloc M>
    class Var {
      public:
        template<typename PrecisionType, typename Container, bool IsVectorBool>
        Var(Matrix<PrecisionType, IsVectorBool> &A, Container *var) {
          You_have_a_precision_conflict <
          typename fl::TypeInfo<Container>::Precision_t, PrecisionType > ();
          Mean<M>(A, &var);
          PrecisionType ep = 0.0;
          PrecisionType va = 0.0;
          index_t n = A.n_rows();
          for (index_t i = 0; i < A.n_cols(); i++) {
            for (index_t j = 0; j < n; j++) {
              PrecisionType c = A.get(j, i) - var[i];
              ep = ep + c;
              va = va + c * c;
            }
            var[i] = (va - ep * ep / n) / (n - 1);
          }
        }
    };

    /**
     * @brief Computes the standard deviation of a vector.
     *
     * @code
     *  fl::la::Matrix<double> a;
     *  fl::la::Matrix<double> vars;
     *  fl::la::Random(30, 23, &a);
     *  fl::la::Std<fl::la::Init>(a, &var);
     *
     * @endcode
     * @param A the input vector
     * @return the standard deviation
     */
    template<fl::la::MemoryAlloc M>
    class Std {
      public:
        template<typename PrecisionType, typename Container, bool IsVectorBool>
        Std(Matrix<PrecisionType, IsVectorBool> &A, Container *sttd) {
          You_have_a_precision_conflict <
          typename fl::TypeInfo<Container>::Precision_t, PrecisionType > ();
          Var<M>(A, &sttd);
          for (index_t i = 0; i < A.n_cols(); i++) {
            sttd[i] = sqrt(sttd[i]);
          }
        }
    };

    /**
     * @brief Computes the sigmoid function of a real number x
     * Sigmoid(x) = 1/[1+exp(-x)]
     *
     * @param x the input real number
     * @return the sigmoid function value
     */
    template<typename PrecisionType>
    static PrecisionType Sigmoid(PrecisionType x) {
      return 1.0 / (1.0 + exp(-x));
    }
    template<typename Archive>
    void serialize(Archive &ar, const unsigned int version) {
    }
}; // class ops

}
} // namsepace fl dense

#endif
